// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `PersistedSession`. These exercise the SQLite
//! roundtrip without touching the user's home directory by using
//! [`PersistedSession::open_in`] with a per-test [`tempfile::TempDir`].

use shirl_core::PersistedSession;
use sweet_core::{MemoryItem, Message, Session, SessionId};

fn user_message(content: &str) -> MemoryItem {
    MemoryItem::Message(Message::user(content))
}

#[test]
fn open_in_creates_intermediate_directories() {
    let base = tempfile::tempdir().unwrap();
    let id = SessionId::new();

    let session = PersistedSession::open_in(base.path(), id.clone()).unwrap();

    assert_eq!(session.id(), &id);
    let db_path = base.path().join(format!("sessions/{}/session.db", id));
    assert!(
        db_path.exists(),
        "expected session.db at {}",
        db_path.display()
    );
}

#[test]
fn pushed_items_persist_across_reopen() {
    let base = tempfile::tempdir().unwrap();
    let id = SessionId::new();
    {
        let mut session = PersistedSession::open_in(base.path(), id.clone()).unwrap();
        session.push(user_message("first")).unwrap();
        session.push(user_message("second")).unwrap();
        assert_eq!(session.items().len(), 2);
    }

    let session = PersistedSession::open_in(base.path(), id.clone()).unwrap();
    let items = session.items();
    assert_eq!(items.len(), 2);
    let (MemoryItem::Message(a), MemoryItem::Message(b)) = (&items[0], &items[1]);
    assert_eq!(a.text_content(), "first");
    assert_eq!(b.text_content(), "second");
}

#[test]
fn clear_empties_session_and_persists() {
    let base = tempfile::tempdir().unwrap();
    let id = SessionId::new();
    {
        let mut session = PersistedSession::open_in(base.path(), id.clone()).unwrap();
        session.push(user_message("doomed")).unwrap();
        session.clear().unwrap();
        assert!(session.items().is_empty());
    }

    let session = PersistedSession::open_in(base.path(), id).unwrap();
    assert!(session.items().is_empty());
}

#[test]
fn replace_range_substitutes_compacted_messages_and_persists() {
    let base = tempfile::tempdir().unwrap();
    let id = SessionId::new();
    {
        let mut session = PersistedSession::open_in(base.path(), id.clone()).unwrap();
        session.push(user_message("a")).unwrap();
        session.push(user_message("b")).unwrap();
        session.push(user_message("c")).unwrap();

        let mut compacted = Message::user("summary");
        compacted.compacted = true;
        session
            .replace_range(0..2, vec![MemoryItem::Message(compacted)])
            .unwrap();
        assert_eq!(session.items().len(), 2);
    }

    let session = PersistedSession::open_in(base.path(), id).unwrap();
    let items = session.items();
    assert_eq!(items.len(), 2);
    let MemoryItem::Message(msg) = &items[0];
    assert_eq!(msg.text_content(), "summary");
    assert!(msg.compacted);
    let MemoryItem::Message(m) = &items[1];
    assert_eq!(m.text_content(), "c");
}
