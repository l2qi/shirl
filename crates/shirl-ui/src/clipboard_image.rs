// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Read images from the system clipboard, re-encode to PNG, and persist to a
//! cache directory so the existing `@path` image-attach pipeline can pick
//! them up.
//!
//! Two cases handled:
//!
//! - **Raw image data** on the clipboard (e.g. macOS screenshot to clipboard
//!   via `Cmd+Ctrl+Shift+4`, or right-click → Copy Image in a browser).
//!   Read via `arboard::Clipboard::get_image()` as an RGBA buffer, then
//!   re-encoded to PNG.
//! - **A file list** of one or more image paths (e.g. Cmd+C on an image file
//!   in macOS Finder, or Ctrl+C in GNOME Files). The first path with a
//!   recognised image extension is loaded via the `image` crate and
//!   re-encoded to PNG.
//!
//! Re-encoding everything to PNG gives us a single MIME type to feed
//! downstream providers regardless of how the image entered the clipboard,
//! and strips any wrapper metadata that providers occasionally reject.
//!
//! The module is feature-gated behind `clipboard-image` (default on). Builds
//! without the feature still compile; [`read_clipboard_png`] always returns
//! [`ClipboardImageError::Unsupported`] in that configuration.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Recognised image extensions when picking a file off a clipboard file list.
#[cfg(feature = "clipboard-image")]
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

/// Subdirectory under `~/.shirl/cache/` where pasted images are stored.
const SUBDIR_PARTS: &[&str] = &[".shirl", "cache", "clipboard"];

/// Error variants surfaced to the UI when a clipboard paste-image attempt fails.
#[derive(Debug, thiserror::Error)]
pub enum ClipboardImageError {
    /// The clipboard does not contain any image data or image file.
    /// Distinct from a hard error — used to render "no image in clipboard".
    #[error("no image in clipboard")]
    NoImage,
    /// No clipboard backend is available (e.g. headless Linux without X11
    /// or Wayland). Distinct from `NoImage` so the message can be more
    /// actionable.
    #[error("no clipboard backend available")]
    NoClipboard,
    /// Clipboard backend open / connection failure that is neither
    /// "platform unsupported" nor a decode error — typically an X11/Wayland
    /// connection problem reported by `arboard::Clipboard::new()`.
    #[error("clipboard backend error: {0}")]
    Backend(String),
    /// Failed to decode the image bytes or file.
    #[error("image decode failed: {0}")]
    Decode(String),
    /// Failed to re-encode the image as PNG.
    #[error("PNG encode failed: {0}")]
    Encode(String),
    /// Filesystem error reading a file from the clipboard file list or
    /// writing the resulting PNG to the cache directory.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// `clipboard-image` feature disabled at build time.
    #[error("clipboard-image feature disabled")]
    Unsupported,
}

/// Read the system clipboard. If it carries image data or a file list whose
/// first entry is a recognised image file, return PNG-encoded bytes.
///
/// Re-encodes to PNG even when the source is already PNG — a uniform output
/// MIME type simplifies the rest of the pipeline and the cost is bounded
/// (one decode + one encode of a screenshot-sized buffer).
#[cfg(feature = "clipboard-image")]
pub fn read_clipboard_png() -> Result<Vec<u8>, ClipboardImageError> {
    let mut clipboard = arboard::Clipboard::new().map_err(|err| match err {
        arboard::Error::ClipboardNotSupported => ClipboardImageError::NoClipboard,
        other => ClipboardImageError::Backend(other.to_string()),
    })?;

    // Try direct image data first — the common case for screenshots and
    // browser "Copy Image". Translate "not available" into the NoImage branch
    // so we can try the file-list fallback before giving up.
    match clipboard.get_image() {
        Ok(image_data) => return encode_rgba_as_png(&image_data),
        Err(arboard::Error::ContentNotAvailable) => {}
        Err(arboard::Error::ClipboardNotSupported) => {
            return Err(ClipboardImageError::NoClipboard);
        }
        Err(other) => return Err(ClipboardImageError::Decode(other.to_string())),
    }

    // Fallback: Finder/Nautilus-style copy of an image file places a file
    // list on the clipboard rather than raw bytes. Find the first path with
    // a recognised image extension and re-encode through `image::open`.
    match clipboard.get().file_list() {
        Ok(paths) => {
            for path in paths {
                if has_image_extension(&path) {
                    return load_and_encode_file(&path);
                }
            }
            Err(ClipboardImageError::NoImage)
        }
        Err(arboard::Error::ContentNotAvailable) => Err(ClipboardImageError::NoImage),
        Err(arboard::Error::ClipboardNotSupported) => Err(ClipboardImageError::NoClipboard),
        Err(other) => Err(ClipboardImageError::Decode(other.to_string())),
    }
}

/// Disabled-feature stub: surfaces a deterministic error so the UI can show
/// a clean message instead of failing at link time.
#[cfg(not(feature = "clipboard-image"))]
pub fn read_clipboard_png() -> Result<Vec<u8>, ClipboardImageError> {
    Err(ClipboardImageError::Unsupported)
}

/// Convert an arboard RGBA buffer to PNG-encoded bytes.
#[cfg(feature = "clipboard-image")]
fn encode_rgba_as_png(data: &arboard::ImageData<'_>) -> Result<Vec<u8>, ClipboardImageError> {
    let width = u32::try_from(data.width)
        .map_err(|_| ClipboardImageError::Decode("image width too large".into()))?;
    let height = u32::try_from(data.height)
        .map_err(|_| ClipboardImageError::Decode("image height too large".into()))?;
    let buf = image::RgbaImage::from_raw(width, height, data.bytes.to_vec())
        .ok_or_else(|| ClipboardImageError::Decode("RGBA buffer size mismatch".into()))?;
    encode_rgba_image_as_png(buf)
}

/// Encode an RGBA8 image as PNG. Shared by the raw-clipboard and file-list
/// paths. Takes the image by value so `DynamicImage::ImageRgba8` can wrap
/// it without an extra full-buffer clone.
#[cfg(feature = "clipboard-image")]
fn encode_rgba_image_as_png(img: image::RgbaImage) -> Result<Vec<u8>, ClipboardImageError> {
    let mut bytes = Vec::with_capacity(img.as_raw().len());
    image::DynamicImage::ImageRgba8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .map_err(|err| ClipboardImageError::Encode(err.to_string()))?;
    Ok(bytes)
}

/// Open an image file from disk and re-encode as PNG.
#[cfg(feature = "clipboard-image")]
fn load_and_encode_file(path: &Path) -> Result<Vec<u8>, ClipboardImageError> {
    let img = image::open(path).map_err(|err| ClipboardImageError::Decode(err.to_string()))?;
    encode_rgba_image_as_png(img.into_rgba8())
}

/// True if the path's extension (lowercased) is one of [`IMAGE_EXTS`].
#[cfg(feature = "clipboard-image")]
fn has_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.as_str()))
}

/// Resolve the default clipboard cache directory: `~/.shirl/cache/clipboard/`.
///
/// Returns `None` if `dirs::home_dir()` returns `None` (extremely rare;
/// happens on misconfigured systems with no HOME). Callers should treat the
/// `None` case as "no cache available" and skip persistence.
pub fn default_cache_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let mut path = home;
    for part in SUBDIR_PARTS {
        path.push(part);
    }
    Some(path)
}

/// Persist PNG bytes to `dir/clip-{nanos}.png` and return the absolute path.
/// Creates the directory on demand.
pub fn save_to_dir(dir: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("clip-{nanos}.png"));
    std::fs::write(&path, bytes)?;
    Ok(path)
}

/// Delete cache entries under `dir` older than `ttl`. Best-effort: errors on
/// individual files are swallowed (only a top-level read failure propagates).
/// Returns the count of files removed. A missing `dir` is a no-op.
pub fn sweep_dir(dir: &Path, ttl: Duration) -> std::io::Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let cutoff = SystemTime::now().checked_sub(ttl).unwrap_or(UNIX_EPOCH);
    let mut removed = 0usize;
    for entry in std::fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let too_old = metadata
            .modified()
            .map(|mtime| mtime < cutoff)
            .unwrap_or(false);
        if too_old && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "clipboard-image")]
    #[test]
    fn has_image_extension_matches_supported_formats() {
        for ext in IMAGE_EXTS {
            assert!(has_image_extension(Path::new(&format!("/tmp/x.{ext}"))));
            assert!(has_image_extension(Path::new(&format!(
                "/tmp/x.{}",
                ext.to_uppercase()
            ))));
        }
    }

    #[cfg(feature = "clipboard-image")]
    #[test]
    fn has_image_extension_rejects_non_image_paths() {
        assert!(!has_image_extension(Path::new("/tmp/notes.txt")));
        assert!(!has_image_extension(Path::new("/tmp/source.rs")));
        assert!(!has_image_extension(Path::new("/tmp/dir")));
        assert!(!has_image_extension(Path::new("/tmp/.hidden")));
        // SVG isn't in the allow list — providers don't accept it.
        assert!(!has_image_extension(Path::new("/tmp/vector.svg")));
    }

    /// Round-trip a tiny RGBA buffer through the encode path. Validates we
    /// produce valid PNG bytes that decode back to the same pixels — covers
    /// the wire-format contract the real paste path depends on.
    #[cfg(feature = "clipboard-image")]
    #[test]
    fn encode_rgba_image_round_trips_through_png() {
        // 2x2 image: red, green, blue, white.
        let raw: Vec<u8> = vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 255, 255, // (1,1) white
        ];
        let img = image::RgbaImage::from_raw(2, 2, raw.clone()).unwrap();
        let png_bytes = encode_rgba_image_as_png(img).expect("encode");
        assert!(png_bytes.starts_with(&[0x89, b'P', b'N', b'G']));
        let decoded = image::load_from_memory(&png_bytes)
            .expect("decode PNG")
            .to_rgba8();
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.into_raw(), raw);
    }

    #[test]
    fn save_to_dir_writes_png_and_returns_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let png = b"\x89PNG\r\n\x1a\nfake".to_vec();
        let path = save_to_dir(tmp.path(), &png).expect("save");
        assert!(path.is_absolute());
        assert!(path.starts_with(tmp.path()));
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert!(name.starts_with("clip-"));
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("png"));
        assert_eq!(std::fs::read(&path).unwrap(), png);
    }

    #[test]
    fn save_to_dir_creates_missing_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        let path = save_to_dir(&nested, b"x").expect("save");
        assert!(path.starts_with(&nested));
        assert!(nested.is_dir());
    }

    #[test]
    fn sweep_dir_zero_ttl_removes_everything() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.png"), b"x").unwrap();
        std::fs::write(tmp.path().join("b.png"), b"x").unwrap();
        let removed = sweep_dir(tmp.path(), Duration::ZERO).unwrap();
        assert_eq!(removed, 2);
        assert!(!tmp.path().join("a.png").exists());
        assert!(!tmp.path().join("b.png").exists());
    }

    #[test]
    fn sweep_dir_long_ttl_keeps_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.png"), b"x").unwrap();
        let removed = sweep_dir(tmp.path(), Duration::from_secs(100 * 365 * 86_400)).unwrap();
        assert_eq!(removed, 0);
        assert!(tmp.path().join("a.png").exists());
    }

    #[test]
    fn sweep_dir_on_missing_dir_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("never-created");
        let removed = sweep_dir(&missing, Duration::from_secs(86_400)).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn default_cache_dir_ends_in_clipboard_subdir() {
        // Sanity-check the segment list. Doesn't assert HOME's value to avoid
        // making the test sensitive to the user's environment.
        let path = default_cache_dir().expect("home dir available in test env");
        let tail: Vec<_> = path
            .components()
            .rev()
            .take(SUBDIR_PARTS.len())
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let expected: Vec<String> = SUBDIR_PARTS.iter().rev().map(|s| s.to_string()).collect();
        assert_eq!(tail, expected);
    }
}
