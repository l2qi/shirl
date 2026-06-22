// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Result};
use shirl_core::{AuthStore, ShirlConfig};
use shirl_llm::catalog::{Catalog, CatalogModel, ReasoningOption};
use std::collections::HashMap;

use shirl_ui::{Command, PickerEntry, PickerRenderState, PickerSection, SharedIo};

/// Compact reasoning-capability hint for a model picker row: the dialect kinds
/// the model supports (`on/off`, `effort`, `budget`), or empty when it exposes
/// no reasoning control. ASCII only.
fn reasoning_hint(options: &[ReasoningOption]) -> String {
    if options.is_empty() {
        return String::new();
    }
    let kinds: Vec<&str> = options.iter().map(ReasoningOption::kind_label).collect();
    format!("  {}", kinds.join("/"))
}

// ---------------------------------------------------------------------------
// Picker state
// ---------------------------------------------------------------------------

/// State backing the interactive model picker.
struct PickerState {
    entries_by_provider: HashMap<String, Vec<CatalogModel>>,
    providers: Vec<String>,
    displayed: Vec<PickerEntry>,
    selected_index: usize,
    filter: String,
    loading: bool,
    current: String,
}

impl PickerState {
    fn new(providers: Vec<String>, current: String) -> Self {
        Self {
            entries_by_provider: HashMap::new(),
            providers,
            displayed: Vec::new(),
            selected_index: 0,
            filter: String::new(),
            loading: true,
            current,
        }
    }

    fn move_selection(&mut self, delta: i32) {
        let total = self.displayed.len();
        if total == 0 {
            return;
        }
        let max = total - 1;
        let new = self.selected_index as i32 + delta;
        self.selected_index = if new < 0 {
            0
        } else if new > max as i32 {
            max
        } else {
            new as usize
        };
    }

    fn selected_entry(&self) -> Option<&PickerEntry> {
        self.displayed.get(self.selected_index)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn fuzzy_match(haystack: &str, needle: &str) -> bool {
    let hay = haystack.to_lowercase();
    let needle = needle.to_lowercase();
    let mut hay_chars = hay.chars();
    for nc in needle.chars() {
        loop {
            match hay_chars.next() {
                Some(hc) if hc == nc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// Truncate `s` to `max` display characters, appending `...` if truncated.
pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let mut truncated: String = chars[..max - 1].iter().collect();
    truncated.push('…');
    truncated
}

pub(crate) fn provider_display_name(
    provider_id: &str,
    catalog: &Catalog,
    config: &ShirlConfig,
) -> String {
    if let Some(custom) = config.providers.get(provider_id) {
        return custom
            .display_name
            .clone()
            .unwrap_or_else(|| provider_id.to_string());
    }
    catalog
        .get_provider(provider_id)
        .map(|p| p.name.clone())
        .unwrap_or_else(|| provider_id.to_string())
}

/// Usable content width for a picker row, derived from the current terminal
/// size. Delegates the geometry to [`shirl_ui::picker_row_width`] so this stays in
/// sync with the popup rendering in `shirl_ui::render_popup`.
pub(crate) fn picker_row_width() -> usize {
    let screen_w = crossterm::terminal::size().map(|(w, _)| w).unwrap_or(80);
    shirl_ui::picker_row_width(screen_w) as usize
}

/// Persist `api_key` for `provider_id` to `auth.toml`.
pub(crate) fn save_provider_key(
    auth: &mut AuthStore,
    provider_id: &str,
    api_key: &str,
) -> Result<()> {
    auth.set(provider_id, api_key);
    auth.save(&AuthStore::default_path()?)
}

// ---------------------------------------------------------------------------
// Entry / section builders
// ---------------------------------------------------------------------------

pub(crate) fn build_provider_entries(
    catalog: &Catalog,
    config: &ShirlConfig,
    auth: &AuthStore,
    max_row_width: usize,
) -> Vec<PickerEntry> {
    // Layout: <id_col> <name_col>. Split the row width roughly in half but
    // give the id a soft minimum so short provider ids stay readable.
    let id_w = (max_row_width / 2).max(10);
    let name_w = max_row_width.saturating_sub(id_w + 1);
    catalog
        .providers
        .iter()
        .map(|p| {
            let display = provider_display_name(&p.id, catalog, config);
            PickerEntry {
                id: p.id.clone(),
                display: format!(
                    "{:<id_w$} {:<name_w$}",
                    truncate_str(&p.id, id_w),
                    truncate_str(&display, name_w),
                ),
                is_current: auth.contains(&p.id),
            }
        })
        .collect()
}

fn build_model_sections(
    state: &PickerState,
    catalog: &Catalog,
    config: &ShirlConfig,
    max_row_width: usize,
) -> Vec<PickerSection> {
    let filter = state.filter.to_lowercase();
    let is_filtering = !filter.is_empty();

    let mut sections = Vec::new();

    for provider_id in &state.providers {
        let models = match state.entries_by_provider.get(provider_id) {
            Some(m) => m,
            None => continue,
        };

        let filtered: Vec<&CatalogModel> = if is_filtering {
            models
                .iter()
                .filter(|m| fuzzy_match(&m.id, &filter) || fuzzy_match(&m.name, &filter))
                .collect()
        } else {
            models.iter().collect()
        };

        if filtered.is_empty() {
            continue;
        }

        let entries: Vec<PickerEntry> = filtered
            .iter()
            .map(|m| {
                let ctx_str = m
                    .context_window
                    .filter(|n| *n > 0)
                    .map(|n| format!("  {}k", n / 1000))
                    .unwrap_or_default();
                // Compact reasoning-capability hint (dialect kinds only; the
                // full per-value detail is shown by `/reasoning`). ASCII so the
                // byte-length width math below stays correct.
                let reason_str = reasoning_hint(&m.reasoning_options);
                let id = format!("{}/{}", provider_id, m.id);
                let is_current = id == state.current;
                // Layout: <id_col> <name_col> <reason><ctx_suffix>
                let suffix = format!("{reason_str}{ctx_str}");
                let ctx_len = suffix.len();
                let available = max_row_width.saturating_sub(ctx_len + 2);
                let id_w = (available * 3 / 5).max(10);
                let name_w = available.saturating_sub(id_w + 1);
                let display = format!(
                    "{:<id_w$} {:<name_w$}{}",
                    truncate_str(&m.id, id_w),
                    truncate_str(&m.name, name_w),
                    suffix,
                );
                PickerEntry {
                    id,
                    display,
                    is_current,
                }
            })
            .collect();

        sections.push(PickerSection {
            header: Some(provider_display_name(provider_id, catalog, config)),
            entries,
        });
    }

    sections
}

fn rebuild_displayed(
    state: &mut PickerState,
    catalog: &Catalog,
    config: &ShirlConfig,
    max_row_width: usize,
) {
    let sections = build_model_sections(state, catalog, config, max_row_width);
    let mut entries = Vec::new();
    for section in &sections {
        entries.extend(section.entries.iter().cloned());
    }
    state.displayed = entries;
    if state.selected_index >= state.displayed.len() {
        state.selected_index = state.displayed.len().saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// Render states
// ---------------------------------------------------------------------------

fn loading_picker_state() -> PickerRenderState {
    PickerRenderState {
        title: "Select model".to_string(),
        sections: vec![],
        selected_index: 0,
        filter: String::new(),
        hint: "Loading models...".to_string(),
        item_label: "models".to_string(),
        input_prefix: "search: ".to_string(),
    }
}

/// Render state for the provider-selection list.
pub(crate) fn provider_picker_render(entries: Vec<PickerEntry>, filter: &str) -> PickerRenderState {
    let hint = if filter.is_empty() {
        "↑↓ navigate · Enter select · Esc cancel"
    } else {
        "↑↓ navigate · Enter select · Esc clear"
    };
    PickerRenderState {
        title: "Select a provider".to_string(),
        sections: vec![PickerSection {
            header: Some("Providers".to_string()),
            entries,
        }],
        selected_index: 0,
        filter: filter.to_string(),
        hint: hint.to_string(),
        item_label: "providers".to_string(),
        input_prefix: "search: ".to_string(),
    }
}

/// Render state for the API-key entry prompt.
pub(crate) fn api_key_render(display: &str, current: &str) -> PickerRenderState {
    PickerRenderState {
        title: format!("Enter API key for {display}"),
        sections: vec![],
        selected_index: 0,
        filter: current.to_string(),
        hint: "Enter to confirm · Esc to cancel".to_string(),
        item_label: String::new(),
        input_prefix: "key: ".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Sync & populate
// ---------------------------------------------------------------------------

async fn sync_picker_render(
    state: &PickerState,
    shared_io: &SharedIo,
    catalog: &Catalog,
    config: &ShirlConfig,
) {
    let max_row_width = picker_row_width();
    let sections = build_model_sections(state, catalog, config, max_row_width);
    let hint = if state.loading {
        "Loading models...".to_string()
    } else if state.displayed.is_empty() {
        "No matches found. Esc to clear".to_string()
    } else if state.filter.is_empty() {
        "↑↓ navigate · Enter select · Esc exit".to_string()
    } else {
        "↑↓ navigate · Enter select · Esc clear".to_string()
    };

    let mut io_guard = shared_io.lock().await;
    io_guard.picker = Some(PickerRenderState {
        title: "Select model".to_string(),
        sections,
        selected_index: state.selected_index,
        filter: state.filter.clone(),
        hint,
        item_label: "models".to_string(),
        input_prefix: "search: ".to_string(),
    });
    let _ = io_guard.draw();
}

async fn populate_model_picker(
    state: &mut PickerState,
    catalog: &Catalog,
    config: &ShirlConfig,
    shared_io: &SharedIo,
) {
    let connected = state.providers.clone();

    for provider_id in &connected {
        let mut models = Vec::new();

        if let Some(cat_provider) = catalog.get_provider(provider_id) {
            models.extend(cat_provider.models.iter().cloned());
        }

        if let Some(extensions) = config.models.get(provider_id) {
            for (id, ext) in extensions {
                if !models.iter().any(|m| m.id == *id) {
                    models.push(CatalogModel {
                        id: id.clone(),
                        name: ext.display_name.clone().unwrap_or_else(|| id.clone()),
                        context_window: ext.context_window,
                        max_output_tokens: None,
                        reasoning: false,
                        vision: false,
                        reasoning_options: Vec::new(),
                    });
                }
            }
        }

        state
            .entries_by_provider
            .insert(provider_id.clone(), models);
    }

    state.loading = false;
    rebuild_displayed(state, catalog, config, picker_row_width());
    sync_picker_render(state, shared_io, catalog, config).await;
}

// ---------------------------------------------------------------------------
// Event loops
// ---------------------------------------------------------------------------

/// Open the model picker, drive it from the input channel, and return the
/// chosen `(provider_id, model_id)`. `Ok(None)` means the user cancelled.
///
/// The popup is opened on entry and closed on every return, so callers never
/// have to manage its lifecycle.
pub(crate) async fn run_model_picker(
    shared_io: &SharedIo,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<Command>,
    catalog: &Catalog,
    config: &ShirlConfig,
    connected: Vec<String>,
    current: String,
) -> Result<Option<(String, String)>> {
    let mut state = PickerState::new(connected, current);

    {
        let mut io_guard = shared_io.lock().await;
        io_guard.open_popup()?;
        io_guard.picker = Some(loading_picker_state());
        let _ = io_guard.draw();
    }

    populate_model_picker(&mut state, catalog, config, shared_io).await;

    loop {
        match cmd_rx.recv().await {
            Some(Command::Submit(_)) => {
                if let Some(entry) = state.selected_entry() {
                    if let Some((provider, model_id)) = entry.id.split_once('/') {
                        let selection = (provider.to_string(), model_id.to_string());
                        let mut io_guard = shared_io.lock().await;
                        io_guard.picker = None;
                        io_guard.close_popup()?;
                        return Ok(Some(selection));
                    }
                }
            }
            Some(Command::Partial(text)) => {
                state.filter = text;
                state.selected_index = 0;
                rebuild_displayed(&mut state, catalog, config, picker_row_width());
                sync_picker_render(&state, shared_io, catalog, config).await;
            }
            Some(Command::SelectMove(delta)) => {
                state.move_selection(delta);
                sync_picker_render(&state, shared_io, catalog, config).await;
            }
            Some(Command::Resize) => {
                rebuild_displayed(&mut state, catalog, config, picker_row_width());
                sync_picker_render(&state, shared_io, catalog, config).await;
            }
            Some(Command::Cancel) | Some(Command::Exit) => {
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = None;
                io_guard.close_popup()?;
                return Ok(None);
            }
            Some(Command::CycleMode)
            | Some(Command::ApprovalKey(_))
            | Some(Command::ToggleTranscript)
            | Some(Command::FilePickerFilter(_))
            | Some(Command::FilePickerAccept)
            | Some(Command::FilePickerClose) => {}
            None => {
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = None;
                let _ = io_guard.close_popup();
                return Err(anyhow::anyhow!("input channel closed"));
            }
        }
    }
}

/// Open the provider list, drive it from the input channel, and return the
/// chosen provider id. `Ok(None)` means the user cancelled.
///
/// The popup is opened on entry and closed on every return.
pub(crate) async fn run_provider_picker(
    shared_io: &SharedIo,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<Command>,
    catalog: &Catalog,
    config: &ShirlConfig,
    auth: &AuthStore,
) -> Result<Option<String>> {
    let mut entries = build_provider_entries(catalog, config, auth, picker_row_width());
    let mut selected_index = 0;
    let mut filter = String::new();
    let mut displayed: Vec<PickerEntry> = entries.clone();

    {
        let mut io_guard = shared_io.lock().await;
        io_guard.open_popup()?;
        io_guard.picker = Some(provider_picker_render(displayed.clone(), ""));
        let _ = io_guard.draw();
    }

    loop {
        match cmd_rx.recv().await {
            Some(Command::Submit(_)) => {
                if let Some(entry) = displayed.get(selected_index) {
                    let id = entry.id.clone();
                    let mut io_guard = shared_io.lock().await;
                    io_guard.picker = None;
                    io_guard.close_popup()?;
                    return Ok(Some(id));
                }
            }
            Some(Command::Partial(text)) => {
                filter = text.to_lowercase();
                displayed = entries
                    .iter()
                    .filter(|e| fuzzy_match(&e.id, &filter) || fuzzy_match(&e.display, &filter))
                    .cloned()
                    .collect();
                selected_index = 0;
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = Some(provider_picker_render(displayed.clone(), &filter));
                let _ = io_guard.draw();
            }
            Some(Command::SelectMove(delta)) => {
                let max = displayed.len().saturating_sub(1) as i32;
                let new = selected_index as i32 + delta;
                selected_index = if new < 0 {
                    0
                } else if new > max {
                    max as usize
                } else {
                    new as usize
                };
                {
                    let mut io_guard = shared_io.lock().await;
                    if let Some(ref mut picker) = io_guard.picker {
                        picker.selected_index = selected_index;
                    }
                    let _ = io_guard.draw();
                }
            }
            Some(Command::Resize) => {
                entries = build_provider_entries(catalog, config, auth, picker_row_width());
                displayed = entries
                    .iter()
                    .filter(|e| fuzzy_match(&e.id, &filter) || fuzzy_match(&e.display, &filter))
                    .cloned()
                    .collect();
                if selected_index >= displayed.len() {
                    selected_index = displayed.len().saturating_sub(1);
                }
                let mut io_guard = shared_io.lock().await;
                let mut render = provider_picker_render(displayed.clone(), &filter);
                render.selected_index = selected_index;
                io_guard.picker = Some(render);
                let _ = io_guard.draw();
            }
            Some(Command::Cancel) | Some(Command::Exit) => {
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = None;
                io_guard.close_popup()?;
                return Ok(None);
            }
            Some(Command::CycleMode)
            | Some(Command::ApprovalKey(_))
            | Some(Command::ToggleTranscript)
            | Some(Command::FilePickerFilter(_))
            | Some(Command::FilePickerAccept)
            | Some(Command::FilePickerClose) => {}
            None => {
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = None;
                let _ = io_guard.close_popup();
                return Err(anyhow::anyhow!("input channel closed"));
            }
        }
    }
}

pub(crate) async fn prompt_api_key(
    shared_io: &SharedIo,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<Command>,
    provider_id: &str,
    catalog: &Catalog,
    config: &ShirlConfig,
) -> Result<Option<String>> {
    let display = provider_display_name(provider_id, catalog, config);

    {
        let mut io_guard = shared_io.lock().await;
        io_guard.open_popup()?;
        io_guard.picker = Some(api_key_render(&display, ""));
        let _ = io_guard.draw();
    }

    let mut filter = String::new();
    loop {
        match cmd_rx.recv().await {
            Some(Command::Submit(_)) => {
                let key = filter.trim().to_string();
                if key.is_empty() {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.picker =
                        Some(api_key_render(&format!("{display} (cannot be empty)"), ""));
                    let _ = io_guard.draw();
                    filter.clear();
                    continue;
                }
                {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.picker = None;
                    io_guard.close_popup()?;
                }
                return Ok(Some(key));
            }
            Some(Command::Partial(text)) => {
                filter = text;
                let mut io_guard = shared_io.lock().await;
                io_guard.picker = Some(api_key_render(&display, &filter));
                let _ = io_guard.draw();
            }
            Some(Command::Cancel) | Some(Command::Exit) => {
                {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.picker = None;
                    io_guard.close_popup()?;
                }
                return Ok(None);
            }
            Some(Command::CycleMode)
            | Some(Command::ApprovalKey(_))
            | Some(Command::ToggleTranscript)
            | Some(Command::FilePickerFilter(_))
            | Some(Command::FilePickerAccept)
            | Some(Command::FilePickerClose) => {}
            None => {
                {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.picker = None;
                    io_guard.close_popup()?;
                }
                return Err(anyhow::anyhow!("input channel closed"));
            }
            _ => {}
        }
    }
}

pub(crate) async fn run_setup_picker(
    shared_io: &SharedIo,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<Command>,
    catalog: &Catalog,
    config: &ShirlConfig,
    auth: &mut AuthStore,
) -> Result<(String, String)> {
    let mut connected: Vec<String> = catalog
        .providers_with_auth(|id| auth.contains(id))
        .iter()
        .map(|p| p.id.clone())
        .collect();

    if connected.is_empty() {
        {
            let mut io_guard = shared_io.lock().await;
            io_guard.insert_lines(&[
                "No providers connected yet. Select a provider to get started.".to_string(),
            ])?;
        }
        let (provider_id, api_key) = loop {
            let provider_id =
                match run_provider_picker(shared_io, cmd_rx, catalog, config, auth).await? {
                    Some(id) => id,
                    None => bail!("setup cancelled"),
                };
            match prompt_api_key(shared_io, cmd_rx, &provider_id, catalog, config).await? {
                Some(key) => break (provider_id, key),
                None => continue,
            }
        };
        save_provider_key(auth, &provider_id, &api_key)?;
        {
            let mut io_guard = shared_io.lock().await;
            let display = provider_display_name(&provider_id, catalog, config);
            io_guard.insert_lines(&[format!("Connected: {} ({})", display, provider_id)])?;
        }
        connected = catalog
            .providers_with_auth(|id| auth.contains(id))
            .iter()
            .map(|p| p.id.clone())
            .collect();
    }

    match run_model_picker(shared_io, cmd_rx, catalog, config, connected, String::new()).await? {
        Some(selection) => Ok(selection),
        None => bail!("setup cancelled"),
    }
}
