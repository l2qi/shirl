// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Resolve `@path/to/media` tokens in user input into content blocks.
//!
//! When the user types `@photo.png` (via the file picker or manually), and the
//! file exists on disk with a recognised image extension, this module keeps the
//! `@path` as text and additionally embeds the file as a `ContentBlock::Image`.
//! When the file is a PDF (`@report.pdf`), it is embedded as a
//! `ContentBlock::File` for providers that support document input. Keeping the
//! path inline means models without vision/document support still see the
//! reference. All other `@path` tokens (source code, text files, etc.) are left
//! as text for the model to read via tool calls.

use std::path::Path;

use sweet_core::message::ContentBlock;

/// Image file extensions we recognise (lowercase, without leading dot).
/// Limited to the formats Anthropic and OpenAI both accept; SVG/BMP/TIFF/ICO
/// would be rejected on the wire.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];

/// Non-image file extensions we recognise for binary passthrough.
/// These are sent as `ContentBlock::File` - the provider API handles
/// parsing natively (e.g. PDF rendering).
///
/// Only file types that providers explicitly accept as document inputs are
/// listed here. Text and source-code files (.rs, .py, .md, etc.) are left as
/// text tokens so the model can read/edit them via tool calls - embedding them
/// as binary would waste tokens, break edit workflows, and be rejected by most
/// provider APIs.
const FILE_EXTENSIONS: &[(&str, &str)] = &[("pdf", "application/pdf")];

/// Sentence punctuation peeled off the end of unquoted `@path` tokens before
/// extension lookup. Without this, `@photo.png,` is seen as having extension
/// `png,` and is left as text. Quoted paths (`@"..."`) are taken verbatim.
const TRAILING_PUNCT: &[char] = &[',', '.', ';', ':', '?', '!', ')', ']', '}'];

/// Soft cap on a single embedded image. Larger files are kept as text and a
/// warning surfaced to the user. Providers also impose their own limits; this
/// guards against accidentally slurping a multi-hundred-MB file into memory.
const MAX_IMAGE_BYTES: u64 = 8 * 1024 * 1024;

/// Soft cap on a single embedded file attachment. Kept equal to the image
/// cap: base64 inflates the payload by ~33%, so an 8 MB raw file becomes
/// ~11 MB encoded. The remaining headroom in the providers' ~32 MB total-
/// request limit is needed for the prompt and conversation history.
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

fn mime_for_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => unreachable!("mime_for_ext called with non-image extension"),
    }
}

fn is_image_ext(ext: &str) -> bool {
    IMAGE_EXTENSIONS.contains(&ext)
}

fn is_file_ext(ext: &str) -> Option<&'static str> {
    FILE_EXTENSIONS
        .iter()
        .find(|(e, _)| e.eq_ignore_ascii_case(ext))
        .map(|(_, mime)| *mime)
}

/// Outcome of resolving `@path` tokens against the filesystem.
pub struct Resolved {
    /// Content blocks interleaving text, images, and file attachments.
    pub blocks: Vec<ContentBlock>,
    /// User-visible warnings (e.g. "image too large, kept as text").
    pub warnings: Vec<String>,
}

/// Extract the path token after `@`, supporting quoted strings with
/// backslash escaping.
///
/// After `@`, if the next character is `"`, reads until the matching
/// unescaped `"`, processing standard escape sequences (`\"` -> `"`,
/// `\\` -> `\`). Returns `(path, after)` where `path` is the unescaped
/// content between the quotes and `after` is everything after the
/// closing quote.
///
/// If the opening `"` is not followed by a closing `"` (unterminated
/// quote), returns `None` so the caller can fall back to
/// whitespace-delimited parsing.
///
/// Iterates by `char`, so paths containing non-ASCII characters
/// (e.g. `@"café.png"`) round-trip correctly.
fn extract_quoted_token(remaining: &str) -> Option<(String, &str)> {
    let mut chars = remaining.char_indices();
    if chars.next().map(|(_, c)| c) != Some('"') {
        return None;
    }
    let mut path = String::new();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '\\' => match chars.next() {
                Some((_, '"')) => path.push('"'),
                Some((_, '\\')) => path.push('\\'),
                Some((_, other)) => {
                    // Unknown escape - keep both characters as-is
                    // so `@"foo\bar.png"` works naturally.
                    path.push('\\');
                    path.push(other);
                }
                None => return None, // Trailing backslash, no close quote.
            },
            '"' => {
                // Closing quote found. `i` is the byte index of `"` (1 byte).
                let after = &remaining[i + 1..];
                return Some((path, after));
            }
            _ => path.push(ch),
        }
    }
    // Ran out of input without a closing quote - unterminated.
    None
}

/// Re-emit a parsed `@token` back to the text buffer, preserving the
/// `@"..."` quoted form (with `"` -> `\"` and `\` -> `\\` re-escaped) when
/// the original token was quoted. Used on miss paths so the user's input
/// round-trips verbatim if no media is embedded.
fn push_token_text(buf: &mut String, token: &str, quoted: bool) {
    buf.push('@');
    if quoted {
        buf.push('"');
        for ch in token.chars() {
            match ch {
                '"' => buf.push_str("\\\""),
                '\\' => buf.push_str("\\\\"),
                _ => buf.push(ch),
            }
        }
        buf.push('"');
    } else {
        buf.push_str(token);
    }
}

/// Parse the input line for `@path` tokens, resolve media files, and return
/// a `Vec<ContentBlock>` that interleaves text, image, and file blocks.
///
/// Image tokens (`.png`, `.jpg`, etc.) are kept as text and followed by a
/// `ContentBlock::Image`. Non-image file tokens (`.pdf`) are kept as text and
/// followed by a `ContentBlock::File`. Unrecognised or missing-file `@path`
/// tokens are preserved as text only (the model can read them with the
/// `read_file` tool).
///
/// Files that exceed the relevant size cap are left as text and a warning is
/// added to the returned `warnings` list.
///
/// Paths may be quoted (`@"path with spaces.png"`) with backslash escaping
/// (`\"` for literal `"`, `\\` for literal `\`). Unquoted paths are
/// delimited by whitespace.
///
/// Returns an error only if a file exists on disk but cannot be read.
/// Missing or non-file paths are silently left as text.
pub fn resolve_media(input: &str, cwd: &Path) -> std::io::Result<Resolved> {
    let mut blocks = Vec::new();
    let mut warnings = Vec::new();
    let mut text_buf = String::new();
    let mut remaining = input;

    while let Some(at_pos) = remaining.find('@') {
        // Push everything before the '@' as text.
        text_buf.push_str(&remaining[..at_pos]);
        remaining = &remaining[at_pos + 1..];

        // Try quoted path first, then fall back to whitespace-delimited.
        // For unquoted tokens we also peel sentence punctuation (`,`, `.`,
        // `?`, ...) off the end so `@a.png,` resolves the same as `@a.png`.
        // The peeled tail is preserved in `trailing` and re-emitted as text
        // immediately after the media (or path) so the sentence reads the
        // same way it was typed.
        let (token, trailing, after, quoted): (String, String, &str, bool) =
            if let Some((tok, after)) = extract_quoted_token(remaining) {
                (tok, String::new(), after, true)
            } else {
                let (raw, after) = match remaining.find(char::is_whitespace) {
                    Some(i) => (&remaining[..i], &remaining[i..]),
                    None => (remaining, ""),
                };
                let kept_len = raw.trim_end_matches(TRAILING_PUNCT).len();
                let (tok, trail) = raw.split_at(kept_len);
                (tok.to_string(), trail.to_string(), after, false)
            };

        if token.is_empty() {
            // Lone '@' at end of string or before whitespace (possibly
            // followed by punctuation we stripped - keep that as text too).
            // A literal `@""` round-trips to itself via push_token_text.
            push_token_text(&mut text_buf, &token, quoted);
            text_buf.push_str(&trailing);
            remaining = after;
            continue;
        }

        let path = Path::new(&token);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());

        if let Some(ref ext) = ext {
            if is_image_ext(ext) {
                let full_path = cwd.join(path);
                if full_path.is_file() {
                    let size = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
                    if size > MAX_IMAGE_BYTES {
                        warnings.push(format!(
                            "⚠ Image {} is {:.1} MB (cap is {:.0} MB) - kept as text",
                            token,
                            size as f64 / 1_048_576.0,
                            MAX_IMAGE_BYTES as f64 / 1_048_576.0,
                        ));
                        push_token_text(&mut text_buf, &token, quoted);
                        text_buf.push_str(&trailing);
                        remaining = after;
                        continue;
                    }
                    let data = std::fs::read(&full_path)?;
                    let media_type = mime_for_ext(ext).to_string();
                    // Keep the referenced path inline so models without vision
                    // support (and the model generally) still see what was
                    // attached; the image block is supplementary.
                    push_token_text(&mut text_buf, &token, quoted);
                    blocks.push(ContentBlock::text(std::mem::take(&mut text_buf)));
                    blocks.push(ContentBlock::Image { data, media_type });
                    text_buf.push_str(&trailing);
                    remaining = after;
                    continue;
                }
            }

            if let Some(mime) = is_file_ext(ext) {
                let full_path = cwd.join(path);
                if full_path.is_file() {
                    let size = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
                    if size > MAX_FILE_BYTES {
                        warnings.push(format!(
                            "⚠ File {} is {:.1} MB (cap is {:.0} MB) - kept as text",
                            token,
                            size as f64 / 1_048_576.0,
                            MAX_FILE_BYTES as f64 / 1_048_576.0,
                        ));
                        push_token_text(&mut text_buf, &token, quoted);
                        text_buf.push_str(&trailing);
                        remaining = after;
                        continue;
                    }
                    let data = std::fs::read(&full_path)?;
                    let filename = full_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file")
                        .to_string();
                    // Keep the referenced path inline so models without
                    // document support (and the model generally) still see what
                    // was attached; the file block is supplementary.
                    push_token_text(&mut text_buf, &token, quoted);
                    blocks.push(ContentBlock::text(std::mem::take(&mut text_buf)));
                    blocks.push(ContentBlock::File {
                        data,
                        media_type: mime.to_string(),
                        filename,
                    });
                    text_buf.push_str(&trailing);
                    remaining = after;
                    continue;
                }
            }
        }

        // Not a recognised media type or file doesn't exist - keep as text.
        // Preserve the quoted form so `@"missing file.png"` round-trips
        // verbatim.
        push_token_text(&mut text_buf, &token, quoted);
        text_buf.push_str(&trailing);
        remaining = after;
    }

    // Flush remaining text.
    text_buf.push_str(remaining);
    if !text_buf.is_empty() {
        blocks.push(ContentBlock::text(text_buf));
    }

    Ok(Resolved { blocks, warnings })
}

/// Check whether a `Vec<ContentBlock>` contains any image blocks.
pub fn has_images(blocks: &[ContentBlock]) -> bool {
    blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }))
}

/// Check whether a `Vec<ContentBlock>` contains any file attachment blocks.
pub fn has_files(blocks: &[ContentBlock]) -> bool {
    blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::File { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn resolve(input: &str, cwd: &Path) -> Vec<ContentBlock> {
        resolve_media(input, cwd).unwrap().blocks
    }

    #[test]
    fn no_media_returns_single_text_block() {
        let blocks = resolve("hello world", Path::new("."));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "hello world");
    }

    #[test]
    fn lone_at_sign_preserved() {
        let blocks = resolve("email@domain.com", Path::new("."));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "email@domain.com");
    }

    #[test]
    fn unrecognised_extension_kept_as_text() {
        let blocks = resolve("@data.bin", Path::new("."));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "@data.bin");
    }

    #[test]
    fn text_extensions_left_as_text() {
        // Source-code and text files are NOT embedded as ContentBlock::File.
        // They stay as text tokens so the model can read/edit them via tools.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("main.rs"), b"fn main() {}").unwrap();
        fs::write(dir.path().join("notes.txt"), b"hello").unwrap();
        fs::write(dir.path().join("README.md"), b"# Hello").unwrap();

        for name in &["main.rs", "notes.txt", "README.md"] {
            let input = format!("@{name}");
            let blocks = resolve(&input, dir.path());
            assert_eq!(blocks.len(), 1, "{name} should produce a single text block");
            assert_eq!(
                blocks[0].as_text().unwrap(),
                &input,
                "{name} should round-trip as text, not embed as File"
            );
        }
    }

    #[test]
    fn image_file_keeps_path_and_embeds_block() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("photo.png");
        fs::write(&img_path, b"\x89PNG\r\n\x1a\n").unwrap();

        // Path text is preserved inline so non-vision models still see it,
        // followed by the supplementary image block.
        let blocks = resolve("@photo.png", dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), "@photo.png");
        assert!(
            matches!(&blocks[1], ContentBlock::Image { data, media_type }
            if data == b"\x89PNG\r\n\x1a\n" && media_type == "image/png")
        );
    }

    #[test]
    fn pdf_file_keeps_path_and_embeds_block() {
        let dir = tempfile::tempdir().unwrap();
        let pdf_path = dir.path().join("report.pdf");
        fs::write(&pdf_path, b"%PDF-1.4 fake").unwrap();

        let blocks = resolve("@report.pdf", dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), "@report.pdf");
        assert!(
            matches!(&blocks[1], ContentBlock::File { data, media_type, filename }
            if data == b"%PDF-1.4 fake"
                && media_type == "application/pdf"
                && filename == "report.pdf")
        );
    }

    #[test]
    fn image_with_surrounding_text() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("shot.jpg");
        fs::write(&img_path, [0xFFu8; 100]).unwrap();

        let blocks = resolve("here is the screenshot @shot.jpg please review", dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0].as_text().unwrap(),
            "here is the screenshot @shot.jpg"
        );
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/jpeg"));
        assert_eq!(blocks[2].as_text().unwrap(), " please review");
    }

    #[test]
    fn pdf_with_surrounding_text() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.pdf"), b"%PDF-1.4").unwrap();

        let blocks = resolve("please review @doc.pdf and advise", dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), "please review @doc.pdf");
        assert!(matches!(&blocks[1], ContentBlock::File { media_type, .. }
            if media_type == "application/pdf"));
        assert_eq!(blocks[2].as_text().unwrap(), " and advise");
    }

    #[test]
    fn missing_file_kept_as_text() {
        let blocks = resolve("@nonexistent.png", Path::new("."));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "@nonexistent.png");
    }

    #[test]
    fn missing_pdf_kept_as_text() {
        let blocks = resolve("@nonexistent.pdf", Path::new("."));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "@nonexistent.pdf");
    }

    #[test]
    fn multiple_images() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.png"), [1u8; 10]).unwrap();
        fs::write(dir.path().join("b.gif"), [2u8; 20]).unwrap();

        let blocks = resolve("@a.png and @b.gif", dir.path());
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].as_text().unwrap(), "@a.png");
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
        assert_eq!(blocks[2].as_text().unwrap(), " and @b.gif");
        assert!(matches!(&blocks[3], ContentBlock::Image { .. }));
    }

    #[test]
    fn mixed_image_and_pdf() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("photo.png"), [1u8; 10]).unwrap();
        fs::write(dir.path().join("doc.pdf"), b"%PDF-1.4").unwrap();

        let blocks = resolve("@photo.png and @doc.pdf ok", dir.path());
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0].as_text().unwrap(), "@photo.png");
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
        assert_eq!(blocks[2].as_text().unwrap(), " and @doc.pdf");
        assert!(matches!(&blocks[3], ContentBlock::File { .. }));
        assert_eq!(blocks[4].as_text().unwrap(), " ok");
    }

    #[test]
    fn has_images_helper() {
        assert!(!has_images(&[ContentBlock::text("hi")]));
        assert!(has_images(&[
            ContentBlock::text("hi"),
            ContentBlock::Image {
                data: vec![],
                media_type: "image/png".to_string(),
            },
        ]));
    }

    #[test]
    fn has_files_helper() {
        assert!(!has_files(&[ContentBlock::text("hi")]));
        assert!(has_files(&[ContentBlock::File {
            data: vec![],
            media_type: "application/pdf".to_string(),
            filename: "doc.pdf".to_string(),
        }]));
        // Images are not files
        assert!(!has_files(&[ContentBlock::Image {
            data: vec![],
            media_type: "image/png".to_string(),
        }]));
    }

    #[test]
    fn subdirectory_image_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("assets")).unwrap();
        fs::write(dir.path().join("assets/logo.webp"), [0u8; 50]).unwrap();

        let blocks = resolve("@assets/logo.webp", dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), "@assets/logo.webp");
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/webp"));
    }

    #[test]
    fn subdirectory_pdf_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("docs")).unwrap();
        fs::write(dir.path().join("docs/report.pdf"), b"%PDF-1.4").unwrap();

        let blocks = resolve("@docs/report.pdf", dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), "@docs/report.pdf");
        assert!(matches!(&blocks[1], ContentBlock::File { media_type, .. }
            if media_type == "application/pdf"));
    }

    #[test]
    fn trailing_punctuation_does_not_block_image() {
        // Reproduces the user-reported bug where typing
        //     "screenshot in @tmp2/a.png, can you ..."
        // sent no image to the model because the path token was
        // `tmp2/a.png,` and Path::extension yielded `Some("png,")`.
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("tmp2")).unwrap();
        fs::write(dir.path().join("tmp2/a.png"), b"\x89PNG\r\n\x1a\n").unwrap();

        let blocks = resolve(
            "screenshot in @tmp2/a.png, can you tell me what it contains?",
            dir.path(),
        );
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), "screenshot in @tmp2/a.png");
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/png"));
        assert_eq!(
            blocks[2].as_text().unwrap(),
            ", can you tell me what it contains?"
        );
    }

    #[test]
    fn trailing_punctuation_does_not_block_pdf() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.pdf"), b"%PDF").unwrap();

        let blocks = resolve("see @doc.pdf, please review", dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), "see @doc.pdf");
        assert!(matches!(&blocks[1], ContentBlock::File { .. }));
        assert_eq!(blocks[2].as_text().unwrap(), ", please review");
    }

    #[test]
    fn trailing_punctuation_variants_all_resolve() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.png"), [1u8; 4]).unwrap();
        for suffix in [",", ".", "?", "!", ";", ":", ")", "]", "}", ",,,"] {
            let input = format!("see @a.png{suffix} done");
            let blocks = resolve(&input, dir.path());
            assert!(
                matches!(&blocks[1], ContentBlock::Image { .. }),
                "input {input:?} did not resolve an image block"
            );
            assert_eq!(blocks[2].as_text().unwrap(), format!("{suffix} done"));
        }
    }

    #[test]
    fn unsupported_extension_kept_as_text() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("vector.svg"), b"<svg/>").unwrap();
        let blocks = resolve("@vector.svg", dir.path());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), "@vector.svg");
    }

    #[test]
    fn oversize_image_kept_as_text_with_warning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.png");
        let big = vec![0u8; (MAX_IMAGE_BYTES + 1) as usize];
        fs::write(&path, &big).unwrap();

        let result = resolve_media("@huge.png", dir.path()).unwrap();
        assert_eq!(result.blocks.len(), 1);
        assert_eq!(result.blocks[0].as_text().unwrap(), "@huge.png");
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("huge.png"));
    }

    #[test]
    fn oversize_pdf_kept_as_text_with_warning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.pdf");
        let big = vec![0u8; (MAX_FILE_BYTES + 1) as usize];
        fs::write(&path, &big).unwrap();

        let result = resolve_media("@huge.pdf", dir.path()).unwrap();
        assert_eq!(result.blocks.len(), 1);
        assert_eq!(result.blocks[0].as_text().unwrap(), "@huge.pdf");
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("huge.pdf"));
    }

    // -----------------------------------------------------------------
    // Quoted path tests
    // -----------------------------------------------------------------

    #[test]
    fn quoted_path_with_spaces_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("my screenshot.png");
        fs::write(&img_path, b"\x89PNG\r\n\x1a\n").unwrap();

        let blocks = resolve(r#"@"my screenshot.png" the rest"#, dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"my screenshot.png""#);
        assert!(
            matches!(&blocks[1], ContentBlock::Image { data, media_type }
            if data == b"\x89PNG\r\n\x1a\n" && media_type == "image/png")
        );
        assert_eq!(blocks[2].as_text().unwrap(), " the rest");
    }

    #[test]
    fn quoted_pdf_with_spaces_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let pdf_path = dir.path().join("my report.pdf");
        fs::write(&pdf_path, b"%PDF-1.4").unwrap();

        let blocks = resolve(r#"@"my report.pdf" the rest"#, dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"my report.pdf""#);
        assert!(
            matches!(&blocks[1], ContentBlock::File { data, media_type, filename }
            if data == b"%PDF-1.4"
                && media_type == "application/pdf"
                && filename == "my report.pdf")
        );
        assert_eq!(blocks[2].as_text().unwrap(), " the rest");
    }

    #[test]
    fn quoted_path_with_surrounding_text() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("screen shot.png");
        fs::write(&img_path, [0xFFu8; 50]).unwrap();

        let blocks = resolve(
            r#"please look at @"screen shot.png" and advise"#,
            dir.path(),
        );
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0].as_text().unwrap(),
            r#"please look at @"screen shot.png""#
        );
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
        assert_eq!(blocks[2].as_text().unwrap(), " and advise");
    }

    #[test]
    fn quoted_path_at_end_of_input() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("spaces in name.png");
        fs::write(&img_path, [1u8; 10]).unwrap();

        let blocks = resolve(r#"@"spaces in name.png""#, dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"spaces in name.png""#);
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
    }

    #[test]
    fn quoted_path_with_escaped_quotes() {
        let dir = tempfile::tempdir().unwrap();
        // File named: she said "hello".png
        let img_path = dir.path().join(r#"she said "hello".png"#);
        fs::write(&img_path, [2u8; 20]).unwrap();

        let blocks = resolve(r#"@"she said \"hello\".png""#, dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"she said \"hello\".png""#);
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
    }

    #[test]
    fn quoted_path_with_escaped_backslash() {
        let dir = tempfile::tempdir().unwrap();
        // `\\` is unescaped to a single `\` inside the parser. The path
        // `path\file.png` won't match a real file on Unix, so the token
        // falls back to text - and the quoted form is preserved verbatim.
        let blocks = resolve(r#"@"path\\file.png""#, dir.path());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"path\\file.png""#);
    }

    #[test]
    fn unterminated_quote_falls_back_to_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("photo.png");
        fs::write(&img_path, [4u8; 10]).unwrap();

        // Opening quote but no closing quote - falls back to whitespace
        // delimiter, so `@"photo.png` is the token (doesn't match the file
        // because of the leading quote).
        let blocks = resolve(r#"@"photo.png"#, dir.path());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"photo.png"#);
    }

    #[test]
    fn mixed_quoted_and_unquoted() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.png"), [1u8; 10]).unwrap();
        let img_path = dir.path().join("spaces b.png");
        fs::write(&img_path, [2u8; 20]).unwrap();

        let blocks = resolve(r#"@a.png and @"spaces b.png" ok"#, dir.path());
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0].as_text().unwrap(), "@a.png");
        assert!(matches!(&blocks[1], ContentBlock::Image { .. }));
        assert_eq!(blocks[2].as_text().unwrap(), r#" and @"spaces b.png""#);
        assert!(matches!(&blocks[3], ContentBlock::Image { .. }));
        assert_eq!(blocks[4].as_text().unwrap(), " ok");
    }

    #[test]
    fn quoted_path_with_directory_spaces() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("my dir")).unwrap();
        let img_path = dir.path().join("my dir").join("photo.png");
        fs::write(&img_path, [5u8; 30]).unwrap();

        let blocks = resolve(r#"@"my dir/photo.png""#, dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"my dir/photo.png""#);
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/png"));
    }

    #[test]
    fn quoted_path_with_non_ascii_resolves() {
        // Regression: byte-iteration of the path string corrupted multi-byte
        // UTF-8 sequences, so `café.png` resolved to a path the OS couldn't
        // find. With char-iteration the file is located and embedded.
        let dir = tempfile::tempdir().unwrap();
        let img_path = dir.path().join("café.png");
        fs::write(&img_path, b"\x89PNG\r\n\x1a\n").unwrap();

        let blocks = resolve(r#"@"café.png" done"#, dir.path());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"café.png""#);
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/png"));
        assert_eq!(blocks[2].as_text().unwrap(), " done");
    }

    #[test]
    fn quoted_path_with_non_ascii_in_subdir() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("스크린샷")).unwrap();
        let img_path = dir.path().join("스크린샷").join("photo.png");
        fs::write(&img_path, [9u8; 30]).unwrap();

        let blocks = resolve(r#"@"스크린샷/photo.png""#, dir.path());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"스크린샷/photo.png""#);
        assert!(matches!(&blocks[1], ContentBlock::Image { media_type, .. }
            if media_type == "image/png"));
    }

    #[test]
    fn quoted_missing_file_roundtrips_quotes() {
        // Round-trip: the user's `@"..."` form is preserved verbatim
        // when the file isn't found, instead of silently dropping quotes.
        let dir = tempfile::tempdir().unwrap();
        let blocks = resolve(r#"see @"missing file.png" please"#, dir.path());
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].as_text().unwrap(),
            r#"see @"missing file.png" please"#
        );
    }

    #[test]
    fn quoted_oversize_image_roundtrips_quotes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big space.png");
        fs::write(&path, vec![0u8; (MAX_IMAGE_BYTES + 1) as usize]).unwrap();

        let result = resolve_media(r#"@"big space.png" ok"#, dir.path()).unwrap();
        assert_eq!(result.blocks.len(), 1);
        assert_eq!(
            result.blocks[0].as_text().unwrap(),
            r#"@"big space.png" ok"#
        );
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("big space.png"));
    }

    #[test]
    fn quoted_missing_with_embedded_quote_roundtrips() {
        // Escape sequences are re-emitted on miss so the literal user
        // text is preserved.
        let dir = tempfile::tempdir().unwrap();
        let blocks = resolve(r#"@"miss \"q\".png" end"#, dir.path());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].as_text().unwrap(), r#"@"miss \"q\".png" end"#);
    }
}
