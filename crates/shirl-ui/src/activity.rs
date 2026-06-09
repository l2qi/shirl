// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Activity indicator: breathing `⏺`, whimsical spinner words, elapsed timer.

use std::fmt::Write;
use std::time::Duration;

use ratatui::style::Color;

/// Full breath cycle of the activity `⏺` (dim → bright → dim), in milliseconds.
/// ~2.5 s reads as a calm "ThinkPad suspend LED" breath; not too eager.
pub(crate) const BREATH_PERIOD_MS: u128 = 2500;

/// Shared accent color for activity indicators.
pub(crate) const ACCENT: Color = Color::Rgb(217, 119, 87);

/// One tool call currently in flight — held in the redrawable viewport so its
/// `⏺` can pulse. Removed and flushed to scrollback when its result arrives.
#[derive(Clone, Debug)]
pub(crate) struct ActiveTool {
    pub id: String,
    pub name: String,
    pub args: String,
}

#[derive(Default, PartialEq)]
pub(crate) enum LastOutput {
    #[default]
    Start,
    ToolCall,
    ToolResult,
    Content,
}

/// Format a `Duration` as a compact elapsed-time string: `5s`, `1m 35s`,
/// `1h 2m 3s`, `1d 1h 2m 3s`. Only non-zero larger units are shown; seconds
/// are always present.
pub(crate) fn format_elapsed(d: Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    let mut s = String::with_capacity(16); // "1d 1h 2m 3s" worst case
    if days > 0 {
        write!(s, "{}d ", days).expect("write to String is infallible");
    }
    if hours > 0 {
        write!(s, "{}h ", hours).expect("write to String is infallible");
    }
    if mins > 0 {
        write!(s, "{}m ", mins).expect("write to String is infallible");
    }
    write!(s, "{}s", secs).expect("write to String is infallible");
    s
}

/// Breath phase in `[0.0, 1.0]` for the activity `⏺`: 0 = dimmest, 1 =
/// brightest. Cosine-shaped so the transitions are smooth at both endpoints
/// (no hard step at the cycle boundary).
pub(crate) fn breath_phase(elapsed: Duration) -> f32 {
    let t = (elapsed.as_millis() % BREATH_PERIOD_MS) as f32 / BREATH_PERIOD_MS as f32;
    // (1 - cos(2π·t)) / 2 — 0 at t=0, 1 at t=0.5, 0 at t=1.
    (1.0 - (t * std::f32::consts::TAU).cos()) * 0.5
}

/// Color of the activity `⏺` at this point in the breath. Lerps from a
/// perceptual mid-grey at the trough up to [`ACCENT`] at the peak.
pub(crate) fn breath_color(elapsed: Duration) -> Color {
    let t = breath_phase(elapsed);
    let r = lerp_u8(90, 217, t);
    let g = lerp_u8(90, 119, t);
    let b = lerp_u8(90, 87, t);
    Color::Rgb(r, g, b)
}

pub(crate) fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let f = a as f32 + (b as f32 - a as f32) * t;
    // Round (not truncate) so that f32 `cos(π)` precision doesn't land the
    // peak at 254 instead of 255.
    f.round().clamp(0.0, 255.0) as u8
}

/// Whimsical activity words for the working indicator — Shirl's answer to
/// Claude Code's spinner verbs. Cute and a little sparkly, with `Shirling` as
/// the self-referential wink (Claude has `Clauding`). Static and curated: the
/// indicator just picks one, never the model.
///
/// Each entry is `(present participle, simple past)`. The live indicator shows
/// the present (`Sparkling…`); the end-of-turn summary reuses the same word in
/// past tense (`Sparkled for 3s.`). Both forms are stored rather than derived
/// because English past tense is irregular (`Weaving` → `Wove`).
const SPINNER_WORDS: &[(&str, &str)] = &[
    ("Sparkling", "Sparkled"),
    ("Twirling", "Twirled"),
    ("Shimmering", "Shimmered"),
    ("Daydreaming", "Daydreamed"),
    ("Doodling", "Doodled"),
    ("Blooming", "Bloomed"),
    ("Swirling", "Swirled"),
    ("Sprinkling", "Sprinkled"),
    ("Glittering", "Glittered"),
    ("Whisking", "Whisked"),
    ("Conjuring", "Conjured"),
    ("Noodling", "Noodled"),
    ("Musing", "Mused"),
    ("Brewing", "Brewed"),
    ("Tinkering", "Tinkered"),
    ("Flourishing", "Flourished"),
    ("Enchanting", "Enchanted"),
    ("Weaving", "Wove"),
    ("Humming", "Hummed"),
    ("Pondering", "Pondered"),
    ("Dreaming", "Dreamed"),
    ("Wondering", "Wondered"),
    ("Frolicking", "Frolicked"),
    ("Marinating", "Marinated"),
    ("Stargazing", "Stargazed"),
    ("Untangling", "Untangled"),
    ("Imagining", "Imagined"),
    ("Fluttering", "Fluttered"),
    ("Bedazzling", "Bedazzled"),
    ("Shirling", "Shirled"),
];

/// The present-participle word for the live indicator. `seed` is chosen once
/// per turn, so the word stays fixed for the whole turn and a fresh one is
/// picked on the next.
pub(crate) fn spinner_word(seed: u64) -> &'static str {
    SPINNER_WORDS[(seed % SPINNER_WORDS.len() as u64) as usize].0
}

/// The simple-past form of the same word [`spinner_word`] picked for `seed` —
/// used for the end-of-turn summary so it matches the word shown while working.
pub(crate) fn spinner_word_past(seed: u64) -> &'static str {
    SPINNER_WORDS[(seed % SPINNER_WORDS.len() as u64) as usize].1
}

/// How many rows the live tool region should occupy given `count` active tools
/// and `available` free rows. Caps at `available`; renders all when it fits.
pub(crate) fn active_tools_render_count(count: usize, available: u16) -> u16 {
    let avail = available as usize;
    if count == 0 || avail == 0 {
        return 0;
    }
    if count <= avail {
        count as u16
    } else {
        avail as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breath_phase_smooth_cycle() {
        let period = BREATH_PERIOD_MS as u64;
        // Dimmest at the cycle boundary.
        assert!(breath_phase(Duration::from_millis(0)) < 0.01);
        assert!(breath_phase(Duration::from_millis(period)) < 0.01);
        // Brightest at the midpoint.
        assert!(breath_phase(Duration::from_millis(period / 2)) > 0.99);
        // Quarter and three-quarter points sit near 0.5 (cosine cross).
        let q = breath_phase(Duration::from_millis(period / 4));
        let tq = breath_phase(Duration::from_millis(period * 3 / 4));
        assert!((q - 0.5).abs() < 0.02);
        assert!((tq - 0.5).abs() < 0.02);
        // Stays in range across many cycles.
        for ms in [0u64, 100, 500, 1234, 5000, 12345, 999_999] {
            let v = breath_phase(Duration::from_millis(ms));
            assert!((0.0..=1.0).contains(&v), "phase {v} out of range at {ms}ms");
        }
    }

    #[test]
    fn breath_color_endpoints_match_lerp() {
        let dim = breath_color(Duration::from_millis(0));
        let bright = breath_color(Duration::from_millis(BREATH_PERIOD_MS as u64 / 2));
        // Dim: theme-agnostic perceptual mid-grey.
        assert_eq!(dim, Color::Rgb(90, 90, 90));
        // Peak: exactly the accent, so the dot matches the static text.
        assert_eq!(bright, Color::Rgb(217, 119, 87));
        assert_eq!(bright, ACCENT);
    }

    #[test]
    fn active_tools_render_count_fits_or_truncates() {
        // No tools or no room → nothing.
        assert_eq!(active_tools_render_count(0, 4), 0);
        assert_eq!(active_tools_render_count(3, 0), 0);
        // Fits exactly: render all.
        assert_eq!(active_tools_render_count(3, 4), 3);
        assert_eq!(active_tools_render_count(4, 4), 4);
        // Overflows: caps at available (caller reserves last row for "+N more").
        assert_eq!(active_tools_render_count(7, 4), 4);
    }

    #[test]
    fn spinner_word_is_fixed_per_seed() {
        // A given seed always maps to the same word (stable for the turn).
        assert_eq!(spinner_word(3), spinner_word(3));
        // Different seeds generally pick different words.
        assert_ne!(spinner_word(0), spinner_word(1));
        // Any seed lands inside the list, including huge ones (no panic).
        assert!(SPINNER_WORDS.iter().any(|w| w.0 == spinner_word(u64::MAX)));
        // Shirl gets her wink, in both tenses.
        assert!(SPINNER_WORDS.contains(&("Shirling", "Shirled")));
    }

    #[test]
    fn spinner_word_past_matches_present_for_same_seed() {
        // The summary word must be the past tense of the very word shown while
        // working — both keyed by the same seed.
        for seed in 0..SPINNER_WORDS.len() as u64 {
            let present = spinner_word(seed);
            let past = spinner_word_past(seed);
            assert!(SPINNER_WORDS.contains(&(present, past)));
        }
    }

    #[test]
    fn format_elapsed_basic() {
        assert_eq!(format_elapsed(Duration::from_secs(0)), "0s");
        assert_eq!(format_elapsed(Duration::from_secs(5)), "5s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
    }

    #[test]
    fn format_elapsed_minutes() {
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_elapsed(Duration::from_secs(95)), "1m 35s");
        assert_eq!(format_elapsed(Duration::from_secs(3599)), "59m 59s");
    }

    #[test]
    fn format_elapsed_hours() {
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1h 0s");
        assert_eq!(format_elapsed(Duration::from_secs(3723)), "1h 2m 3s");
    }

    #[test]
    fn format_elapsed_days() {
        assert_eq!(format_elapsed(Duration::from_secs(86400)), "1d 0s");
        assert_eq!(format_elapsed(Duration::from_secs(90123)), "1d 1h 2m 3s");
    }
}
