// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_agent::{Capability, CapabilityProvider, CommandContext, CommandHandler, CommandSpec};
use sweet_core::{async_trait, Result};

use crate::compaction::{compact_session, DEFAULT_PRESERVE_RECENT};
use crate::session::PersistedSession;

pub fn parse_slash_command(input: &str) -> Option<(&str, &str)> {
    let trimmed = input.trim();
    let without_slash = trimmed.strip_prefix('/')?;
    if without_slash.is_empty() {
        return None;
    }
    Some(match without_slash.split_once(' ') {
        Some((name, args)) => (name, args.trim()),
        None => (without_slash, ""),
    })
}

pub struct New;

impl CapabilityProvider for New {
    fn id(&self) -> &str {
        "shirl:new"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Command(CommandSpec::new(
            "new",
            "Start a new session",
            "/new",
            New,
        ))]
    }
}

#[async_trait]
impl CommandHandler for New {
    async fn handle(&self, _args: &str, ctx: &mut dyn CommandContext) -> Result<Option<String>> {
        ctx.replace_session(Box::new(PersistedSession::new()?));
        Ok(Some(format!(
            "[new session started: {}]",
            ctx.session().id()
        )))
    }
}

pub struct Clear;

impl CapabilityProvider for Clear {
    fn id(&self) -> &str {
        "shirl:clear"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Command(CommandSpec::new(
            "clear",
            "Clear the current session",
            "/clear",
            Clear,
        ))]
    }
}

#[async_trait]
impl CommandHandler for Clear {
    async fn handle(&self, _args: &str, ctx: &mut dyn CommandContext) -> Result<Option<String>> {
        ctx.session_mut().clear()?;
        Ok(Some(format!("[session cleared: {}]", ctx.session().id())))
    }
}

pub struct Compact;

impl CapabilityProvider for Compact {
    fn id(&self) -> &str {
        "shirl:compact"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Command(CommandSpec::new(
            "compact",
            "Compact session history with an optional hint",
            "/compact [hint]",
            Compact,
        ))]
    }
}

#[async_trait]
impl CommandHandler for Compact {
    async fn handle(&self, args: &str, ctx: &mut dyn CommandContext) -> Result<Option<String>> {
        let hint = if args.trim().is_empty() {
            None
        } else {
            Some(args.trim())
        };
        compact_session(ctx, DEFAULT_PRESERVE_RECENT, hint).await?;
        Ok(Some(format!("[session compacted: {}]", ctx.session().id())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_agent::test_util::MockModel;
    use sweet_agent::{Agent, CommandRouter, ExtensionRegistry};
    use sweet_core::MemoryItem;

    #[test]
    fn parses_slash_command_name_and_args() {
        assert_eq!(parse_slash_command("/clear"), Some(("clear", "")));
        assert_eq!(
            parse_slash_command("  /compact keep decisions  "),
            Some(("compact", "keep decisions"))
        );
        assert_eq!(parse_slash_command("plain text"), None);
        assert_eq!(parse_slash_command("/"), None);
    }

    #[test]
    fn built_ins_produce_command_capabilities() {
        let commands = [
            New.capabilities(),
            Clear.capabilities(),
            Compact.capabilities(),
        ]
        .into_iter()
        .flatten()
        .filter_map(|capability| match capability {
            Capability::Command(command) => Some(command.name),
            _ => None,
        })
        .collect::<Vec<_>>();

        assert_eq!(commands, vec!["new", "clear", "compact"]);
    }

    #[tokio::test]
    async fn built_in_command_registers_through_extension_capability() {
        let mut extensions = ExtensionRegistry::new();
        extensions.register(Clear);
        let router = CommandRouter::from_extension_registry(&extensions);
        let mut agent = Agent::new(MockModel::with_replies(["ok"]));
        agent.step("hi").await.unwrap();

        let result = router.handle("clear", "", &mut agent).await.unwrap();

        assert!(agent.session().messages().is_empty());
        assert!(result.unwrap().starts_with("[session cleared: "));
    }

    #[tokio::test]
    async fn compact_command_uses_command_context() {
        let mut extensions = ExtensionRegistry::new();
        extensions.register(Compact);
        let router = CommandRouter::from_extension_registry(&extensions);
        let mut agent = Agent::new(MockModel::with_replies(["a1", "a2", "a3", "a4", "summary"]));
        agent.step("u1").await.unwrap();
        agent.step("u2").await.unwrap();
        agent.step("u3").await.unwrap();
        agent.step("u4").await.unwrap();
        assert_eq!(agent.session().messages().len(), 8);

        let result = router
            .handle("compact", "keep architecture notes", &mut agent)
            .await
            .unwrap();

        assert!(result.unwrap().starts_with("[session compacted: "));
        assert_eq!(agent.session().items().len(), 8);
        let MemoryItem::Message(msg) = &agent.session().items()[0];
        assert!(msg.compacted);
        let MemoryItem::Message(msg) = &agent.session().items()[1];
        assert!(msg.compacted);
    }

    #[tokio::test]
    async fn unknown_slash_command_is_not_handled() {
        let extensions = ExtensionRegistry::new();
        let router = CommandRouter::from_extension_registry(&extensions);
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));
        let (name, args) = parse_slash_command("/missing arg").unwrap();

        let result = router.handle(name, args, &mut agent).await.unwrap();

        assert_eq!(result, None);
    }
}
