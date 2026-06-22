// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use shirl_core::AuthStore;
use tokio::sync::Mutex;

use shirl_ui::ReplIo;

/// Flatten every connected MCP provider's tools into one spec list. Every
/// agent (main, plan, review) receives the same MCP toolset.
pub(crate) fn flatten_mcp_specs(providers: &[sweet_mcp::McpProvider]) -> Vec<sweet_core::ToolSpec> {
    providers
        .iter()
        .flat_map(|p| p.specs().iter().cloned())
        .collect()
}

fn mcp_config_path() -> Result<std::path::PathBuf> {
    Ok(shirl_core::config_home()?.join("mcp.json"))
}

/// Per-server cap on the MCP connection handshake (transport + tool listing).
const MCP_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Connect to all configured MCP servers, collecting status messages for
/// deferred display. Returns `(providers, status_messages)`.
///
/// This is the shared core logic used by both interactive and headless MCP
/// loaders. Status messages are collected rather than emitted immediately so
/// the caller can choose the output sink (TUI vs stderr).
async fn connect_mcp_servers(
    config: &sweet_mcp::McpConfig,
) -> (Vec<sweet_mcp::McpProvider>, Vec<String>) {
    let mut providers = Vec::new();
    let mut status = Vec::new();

    for (name, server) in &config.servers {
        let filter = sweet_mcp::ToolFilter::new(
            server.allow_tools.clone().unwrap_or_default(),
            server.block_tools.clone().unwrap_or_default(),
        );

        if server.is_stdio() && server.is_http() {
            status.push(format!(
                "MCP: '{name}' sets both `command` and `url`; using stdio"
            ));
        }

        if !server.is_stdio() && !server.is_http() {
            status.push(format!("MCP: skipping '{name}' - no `command` or `url`"));
            continue;
        }

        let connect = async {
            if server.is_stdio() {
                let cmd = server
                    .command
                    .as_deref()
                    .expect("is_stdio() guarantees command is set");
                let args = server.args.clone().unwrap_or_default();
                let env = server.env.clone().unwrap_or_default();
                sweet_mcp::McpProvider::connect_stdio(name, cmd, &args, &env, &filter).await
            } else {
                let url = server
                    .url
                    .as_deref()
                    .expect("is_http() guarantees url is set");
                let headers = server.headers.clone().unwrap_or_default();
                sweet_mcp::McpProvider::connect_http(name, url, &headers, &filter).await
            }
        };

        // Bound the handshake: a slow or wedged server must not hang startup.
        match tokio::time::timeout(MCP_CONNECT_TIMEOUT, connect).await {
            Ok(Ok(provider)) => {
                let tool_count = provider.specs().len();
                providers.push(provider);
                status.push(format!("MCP: connected to '{name}' ({tool_count} tools)"));
            }
            Ok(Err(e)) => {
                status.push(format!("MCP: failed to connect to '{name}': {e}"));
            }
            Err(_) => {
                status.push(format!(
                    "MCP: skipping '{name}' - connection timed out after {}s",
                    MCP_CONNECT_TIMEOUT.as_secs()
                ));
            }
        }
    }

    (providers, status)
}

/// Load and connect MCP providers for interactive mode, emitting status
/// messages via the TUI.
pub(crate) async fn load_mcp_providers(
    io: &Arc<Mutex<ReplIo>>,
    auth: &Mutex<AuthStore>,
) -> Vec<sweet_mcp::McpProvider> {
    let path = match mcp_config_path() {
        Ok(p) => p,
        Err(e) => {
            let mut guard = io.lock().await;
            let _ = guard.insert_lines(&[format!("MCP: skipping - {e}")]);
            return vec![];
        }
    };

    if !path.exists() {
        return vec![];
    }

    let config = match sweet_mcp::McpConfig::from_file(&path) {
        Ok(c) => c,
        Err(e) => {
            let mut guard = io.lock().await;
            let _ = guard.insert_lines(&[format!("MCP: failed to parse {}: {e}", path.display())]);
            return vec![];
        }
    };

    let mcp_env = {
        let auth_guard = auth.lock().await;
        auth_guard.mcp.clone()
    };
    let config = config.resolve_env_vars(&mcp_env);

    let (providers, messages) = connect_mcp_servers(&config).await;
    if !messages.is_empty() {
        let mut guard = io.lock().await;
        let _ = guard.insert_lines(&messages);
    }
    providers
}

/// Load and connect MCP providers for headless mode, emitting status
/// messages to stderr.
pub(crate) async fn load_mcp_providers_headless(auth: &AuthStore) -> Vec<sweet_mcp::McpProvider> {
    let path = match mcp_config_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("MCP: skipping - {e}");
            return vec![];
        }
    };

    if !path.exists() {
        return vec![];
    }

    let config = match sweet_mcp::McpConfig::from_file(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("MCP: failed to parse {}: {e}", path.display());
            return vec![];
        }
    };

    let mcp_env = auth.mcp.clone();
    let config = config.resolve_env_vars(&mcp_env);

    let (providers, messages) = connect_mcp_servers(&config).await;
    for msg in &messages {
        eprintln!("{msg}");
    }
    providers
}
