//! Archive handler supporting ZIP, TAR, and GZIP formats.
//!
//! Recursively unpacks archives up to a configurable depth limit, enforces a
//! maximum decompressed size, and detects zip bombs by monitoring compression
//! ratios.

use guardian_common::{FeatureVector, FileType, GuardianError, Result};
use std::io::{Cursor, Read};

use super::{FileParser, ParsedFile, shannon_entropy};

// ── Limits ──────────────────────────────────────────────────────────────────

/// Maximum recursion depth for nested archives.
const MAX_RECURSION_DEPTH: u32 = 5;

/// Maximum total decompressed size (500 MB).
const MAX_DECOMPRESSED_SIZE: u64 = 500 * 1024 * 1024;

/// Compression ratio above which we flag the archive as a potential zip bomb.
const MAX_COMPRESSION_RATIO: f64 = 100.0;

// ── Archive content structures ──────────────────────────────────────────────

/// Metadata for a single file entry within an archive.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    /// Path / name within the archive.
    pub name: String,
    /// Compressed size in bytes.
    pub compressed_size: u64,
    /// Uncompressed size in bytes (if known).
    pub uncompressed_size: u64,
    /// Detected file type of the entry.
    pub file_type: FileType,
    /// Shannon entropy of the decompressed content.
    pub entropy: f64,
    /// True if this entry is itself an archive (nested).
    pub is_nested_archive: bool,
}

/// Archive format detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    Zip,
    Tar,
    Gzip,
    TarGzip,
}

/// Structured information extracted from an archive.
#[derive(Debug, Clone)]
pub struct ArchiveContents {
    /// Detected archive format.
    pub format: ArchiveFormat,
    /// Total compressed size.
    pub compressed_size: u64,
    /// Total decompressed size.
    pub decompressed_size: u64,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Whether a zip bomb was detected.
    pub is_zip_bomb: bool,
    /// Number of entries.
    pub entry_count: usize,
    /// Recursion depth reached.
    pub max_depth_reached: u32,
    /// Entries in the archive (up to a reasonable limit).
    pub entries: Vec<ArchiveEntry>,
    /// Overall entropy of the raw archive data.
    pub file_entropy: f64,
}

/// Archive format parser.
pub struct ArchiveParser;

impl FileParser for ArchiveParser {
    fn parse(data: &[u8]) -> Result<ParsedFile> {
        let info = parse_archive(data, 0)?;
        Ok(ParsedFile::Archive(info))
    }

    fn extract_features(data: &[u8]) -> Result<FeatureVector> {
        let info = parse_archive(data, 0)?;
        Ok(build_feature_vector(&info, data))
    }
}

// ── Format detection ────────────────────────────────────────────────────────

fn detect_archive_format(data: &[u8]) -> Result<ArchiveFormat> {
    if data.len() < 4 {
        return Err(GuardianError::Parse(
            "Archive: data too small to detect format".to_string(),
        ));
    }

    // ZIP: PK\x03\x04
    if data[0] == 0x50 && data[1] == 0x4B && data[2] == 0x03 && data[3] == 0x04 {
        return Ok(ArchiveFormat::Zip);
    }

    // GZIP: \x1F\x8B
    if data[0] == 0x1F && data[1] == 0x8B {
        return Ok(ArchiveFormat::Gzip);
    }

    // TAR: check for "ustar" at offset 257 (POSIX tar).
    if data.len() > 262 && &data[257..262] == b"ustar" {
        return Ok(ArchiveFormat::Tar);
    }

    // Could still be an uncompressed tar; try heuristic: first entry
    // usually has a name in bytes 0..100 with printable characters.
    if data.len() >= 512 {
        let name_block = &data[0..100];
        let printable = name_block
            .iter()
            .take_while(|&&b| b != 0)
            .all(|&b| b.is_ascii_graphic() || b == b' ' || b == b'/' || b == b'.');
        let has_name = name_block.iter().any(|&b| b.is_ascii_alphanumeric());
        if printable && has_name {
            return Ok(ArchiveFormat::Tar);
        }
    }

    Err(GuardianError::Parse(
        "Archive: unrecognised archive format".to_string(),
    ))
}

// ── Core parsing logic ──────────────────────────────────────────────────────

fn parse_archive(data: &[u8], depth: u32) -> Result<ArchiveContents> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(GuardianError::Parse(format!(
            "Archive: maximum recursion depth ({MAX_RECURSION_DEPTH}) exceeded"
        )));
    }

    let format = detect_archive_format(data)?;
    let file_entropy = shannon_entropy(data);
    let _compressed_size = data.len() as u64;

    match format {
        ArchiveFormat::Zip => parse_zip(data, depth, file_entropy),
        ArchiveFormat::Gzip => parse_gzip(data, depth, file_entropy),
        ArchiveFormat::Tar => parse_tar(data, depth, file_entropy),
        ArchiveFormat::TarGzip => parse_gzip(data, depth, file_entropy),
    }
}

// ── ZIP parser ──────────────────────────────────────────────────────────────

fn parse_zip(data: &[u8], depth: u32, file_entropy: f64) -> Result<ArchiveContents> {
    let reader = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| {
        GuardianError::Parse(format!("ZIP parse failed: {e}"))
    })?;

    let compressed_size = data.len() as u64;
    let mut total_decompressed: u64 = 0;
    let mut entries = Vec::new();
    let mut max_depth_reached = depth;

    for i in 0..archive.len() {
        let file = archive.by_index(i).map_err(|e| {
            GuardianError::Parse(format!("ZIP entry {i} error: {e}"))
        })?;

        if file.is_dir() {
            continue;
        }

        let entry_name = file.name().to_string();
        let compressed_entry = file.compressed_size();
        let uncompressed_entry = file.size();

        // Check decompressed size limit before reading.
        total_decompressed += uncompressed_entry;
        if total_decompressed > MAX_DECOMPRESSED_SIZE {
            return Err(GuardianError::FileTooLarge {
                size: total_decompressed,
                max: MAX_DECOMPRESSED_SIZE,
            });
        }

        // Check compression ratio per entry.
        if compressed_entry > 0 {
            let ratio = uncompressed_entry as f64 / compressed_entry as f64;
            if ratio > MAX_COMPRESSION_RATIO {
                return Err(GuardianError::ArchiveBomb { ratio });
            }
        }

        // Read a limited amount for type detection and entropy.
        let mut buf = Vec::with_capacity(uncompressed_entry.min(1024 * 1024) as usize);
        file.take(1024 * 1024).read_to_end(&mut buf).map_err(|e| {
            GuardianError::Parse(format!("ZIP read error for '{entry_name}': {e}"))
        })?;

        let file_type = FileType::from_magic(&buf);
        let entropy = shannon_entropy(&buf);
        let is_nested = file_type == FileType::Archive;

        entries.push(ArchiveEntry {
            name: entry_name,
            compressed_size: compressed_entry,
            uncompressed_size: uncompressed_entry,
            file_type,
            entropy,
            is_nested_archive: is_nested,
        });

        // Recursively process nested archives (if full content was read).
        if is_nested && buf.len() as u64 == uncompressed_entry {
            if let Ok(nested) = parse_archive(&buf, depth + 1) {
                if nested.max_depth_reached > max_depth_reached {
                    max_depth_reached = nested.max_depth_reached;
                }
            }
        }
    }

    let compression_ratio = if compressed_size > 0 {
        total_decompressed as f64 / compressed_size as f64
    } else {
        0.0
    };

    let is_zip_bomb = compression_ratio > MAX_COMPRESSION_RATIO;

    if is_zip_bomb {
        return Err(GuardianError::ArchiveBomb {
            ratio: compression_ratio,
        });
    }

    Ok(ArchiveContents {
        format: ArchiveFormat::Zip,
        compressed_size,
        decompressed_size: total_decompressed,
        compression_ratio,
        is_zip_bomb,
        entry_count: entries.len(),
        max_depth_reached,
        entries,
        file_entropy,
    })
}

// ── GZIP parser ─────────────────────────────────────────────────────────────

fn parse_gzip(data: &[u8], depth: u32, file_entropy: f64) -> Result<ArchiveContents> {
    let compressed_size = data.len() as u64;

    let mut decoder = flate2::read::GzDecoder::new(Cursor::new(data));
    let mut decompressed = Vec::new();

    // Read up to MAX_DECOMPRESSED_SIZE + 1 to detect oversize.
    let mut limited = (&mut decoder).take(MAX_DECOMPRESSED_SIZE + 1);
    limited.read_to_end(&mut decompressed).map_err(|e| {
        GuardianError::Parse(format!("GZIP decompress failed: {e}"))
    })?;

    if decompressed.len() as u64 > MAX_DECOMPRESSED_SIZE {
        return Err(GuardianError::FileTooLarge {
            size: decompressed.len() as u64,
            max: MAX_DECOMPRESSED_SIZE,
        });
    }

    let decompressed_size = decompressed.len() as u64;

    // Check compression ratio.
    let compression_ratio = if compressed_size > 0 {
        decompressed_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if compression_ratio > MAX_COMPRESSION_RATIO {
        return Err(GuardianError::ArchiveBomb {
            ratio: compression_ratio,
        });
    }

    // Check if the decompressed content is a tar archive.
    let inner_type = FileType::from_magic(&decompressed);
    let is_tar_inside = decompressed.len() > 262 && &decompressed[257..262] == b"ustar";

    let mut entries = Vec::new();
    let mut max_depth_reached = depth;

    if is_tar_inside {
        // It is a .tar.gz — recurse into the tar.
        match parse_tar(&decompressed, depth + 1, shannon_entropy(&decompressed)) {
            Ok(tar_contents) => {
                if tar_contents.max_depth_reached > max_depth_reached {
                    max_depth_reached = tar_contents.max_depth_reached;
                }
                entries = tar_contents.entries;
            }
            Err(_) => {
                // If tar parsing fails, just record the inner blob.
                entries.push(ArchiveEntry {
                    name: "decompressed".to_string(),
                    compressed_size,
                    uncompressed_size: decompressed_size,
                    file_type: inner_type,
                    entropy: shannon_entropy(&decompressed),
                    is_nested_archive: inner_type == FileType::Archive,
                });
            }
        }
    } else {
        entries.push(ArchiveEntry {
            name: "decompressed".to_string(),
            compressed_size,
            uncompressed_size: decompressed_size,
            file_type: inner_type,
            entropy: shannon_entropy(&decompressed),
            is_nested_archive: inner_type == FileType::Archive,
        });

        // Recurse into nested archives.
        if inner_type == FileType::Archive {
            if let Ok(nested) = parse_archive(&decompressed, depth + 1) {
                if nested.max_depth_reached > max_depth_reached {
                    max_depth_reached = nested.max_depth_reached;
                }
            }
        }
    }

    let format = if is_tar_inside {
        ArchiveFormat::TarGzip
    } else {
        ArchiveFormat::Gzip
    };

    Ok(ArchiveContents {
        format,
        compressed_size,
        decompressed_size,
        compression_ratio,
        is_zip_bomb: false,
        entry_count: entries.len(),
        max_depth_reached,
        entries,
        file_entropy,
    })
}

// ── TAR parser ──────────────────────────────────────────────────────────────

fn parse_tar(data: &[u8], depth: u32, file_entropy: f64) -> Result<ArchiveContents> {
    let compressed_size = data.len() as u64;
    let reader = Cursor::new(data);
    let mut archive = tar::Archive::new(reader);

    let mut entries = Vec::new();
    let mut total_decompressed: u64 = 0;
    let mut max_depth_reached = depth;

    let iter = archive.entries().map_err(|e| {
        GuardianError::Parse(format!("TAR parse failed: {e}"))
    })?;

    for entry_result in iter {
        let entry = entry_result.map_err(|e| {
            GuardianError::Parse(format!("TAR entry error: {e}"))
        })?;

        let header = entry.header();
        let entry_size = header.size().unwrap_or(0);
        let entry_name = entry
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());

        // Skip directories.
        if header.entry_type().is_dir() {
            continue;
        }

        total_decompressed += entry_size;
        if total_decompressed > MAX_DECOMPRESSED_SIZE {
            return Err(GuardianError::FileTooLarge {
                size: total_decompressed,
                max: MAX_DECOMPRESSED_SIZE,
            });
        }

        // Read limited content for analysis.
        let mut buf = Vec::with_capacity(entry_size.min(1024 * 1024) as usize);
        entry.take(1024 * 1024).read_to_end(&mut buf).map_err(|e| {
            GuardianError::Parse(format!("TAR read error for '{entry_name}': {e}"))
        })?;

        let file_type = FileType::from_magic(&buf);
        let entropy = shannon_entropy(&buf);
        let is_nested = file_type == FileType::Archive;

        entries.push(ArchiveEntry {
            name: entry_name,
            compressed_size: entry_size, // tar is uncompressed, so same
            uncompressed_size: entry_size,
            file_type,
            entropy,
            is_nested_archive: is_nested,
        });

        // Recurse into nested archives.
        if is_nested && buf.len() as u64 == entry_size {
            if let Ok(nested) = parse_archive(&buf, depth + 1) {
                if nested.max_depth_reached > max_depth_reached {
                    max_depth_reached = nested.max_depth_reached;
                }
            }
        }
    }

    // For tar, compression ratio is ~1:1.
    let compression_ratio = if compressed_size > 0 {
        total_decompressed as f64 / compressed_size as f64
    } else {
        0.0
    };

    Ok(ArchiveContents {
        format: ArchiveFormat::Tar,
        compressed_size,
        decompressed_size: total_decompressed,
        compression_ratio,
        is_zip_bomb: false,
        entry_count: entries.len(),
        max_depth_reached,
        entries,
        file_entropy,
    })
}

// ── Feature vector construction ─────────────────────────────────────────────

fn build_feature_vector(info: &ArchiveContents, data: &[u8]) -> FeatureVector {
    let mut numeric: Vec<f64> = Vec::with_capacity(12);

    numeric.push(info.compressed_size as f64);
    numeric.push(info.decompressed_size as f64);
    numeric.push(info.compression_ratio);
    numeric.push(if info.is_zip_bomb { 1.0 } else { 0.0 });
    numeric.push(info.entry_count as f64);
    numeric.push(info.max_depth_reached as f64);
    numeric.push(info.file_entropy);
    numeric.push(data.len() as f64);

    // Count of nested archives.
    let nested_count = info
        .entries
        .iter()
        .filter(|e| e.is_nested_archive)
        .count();
    numeric.push(nested_count as f64);

    // Average entry entropy.
    let avg_entropy = if info.entries.is_empty() {
        0.0
    } else {
        info.entries.iter().map(|e| e.entropy).sum::<f64>()
            / info.entries.len() as f64
    };
    numeric.push(avg_entropy);

    // Categorical: entry names and file types.
    let mut categorical: Vec<String> =
        info.entries.iter().map(|e| e.name.clone()).collect();
    categorical.push(format!("{:?}", info.format));

    // Metadata.
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "format".to_string(),
        serde_json::json!(format!("{:?}", info.format)),
    );
    metadata.insert(
        "compression_ratio".to_string(),
        serde_json::json!(info.compression_ratio),
    );
    metadata.insert(
        "entry_count".to_string(),
        serde_json::json!(info.entry_count),
    );
    metadata.insert(
        "nested_archives".to_string(),
        serde_json::json!(nested_count),
    );

    FeatureVector {
        numeric,
        categorical,
        metadata,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create an in-memory ZIP with the given entries (name, content).
    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        for (name, content) in entries {
            writer.start_file(*name, options).unwrap();
            writer.write_all(content).unwrap();
        }

        writer.finish().unwrap().into_inner()
    }

    /// Create an in-memory GZIP blob.
    fn make_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::default(),
        );
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Create an in-memory TAR with a single entry.
    fn make_tar(name: &str, content: &[u8]) -> Vec<u8> {
        let buf = Vec::new();
        let mut builder = tar::Builder::new(buf);

        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();

        builder
            .append_data(&mut header, name, Cursor::new(content))
            .unwrap();
        builder.finish().unwrap();
        builder.into_inner().unwrap()
    }

    #[test]
    fn test_parse_zip_basic() {
        let data = make_zip(&[
            ("hello.txt", b"Hello, world!"),
            ("data.bin", &[0xDE, 0xAD, 0xBE, 0xEF]),
        ]);
        let result = ArchiveParser::parse(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        if let Ok(ParsedFile::Archive(info)) = result {
            assert_eq!(info.format, ArchiveFormat::Zip);
            assert_eq!(info.entry_count, 2);
            assert!(!info.is_zip_bomb);
        } else {
            panic!("expected Archive variant");
        }
    }

    #[test]
    fn test_parse_gzip_basic() {
        let content = b"Hello, this is some test content for gzip.";
        let data = make_gzip(content);
        let result = ArchiveParser::parse(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        if let Ok(ParsedFile::Archive(info)) = result {
            assert_eq!(info.format, ArchiveFormat::Gzip);
            assert_eq!(info.decompressed_size, content.len() as u64);
        } else {
            panic!("expected Archive variant");
        }
    }

    #[test]
    fn test_parse_tar_basic() {
        let content = b"tar file content here";
        let data = make_tar("test.txt", content);
        let result = parse_archive(&data, 0);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let info = result.unwrap();
        assert_eq!(info.format, ArchiveFormat::Tar);
        assert_eq!(info.entry_count, 1);
        assert_eq!(info.entries[0].name, "test.txt");
    }

    #[test]
    fn test_parse_tar_gz() {
        let content = b"inner tar content";
        let tar_data = make_tar("inner.txt", content);
        let gz_data = make_gzip(&tar_data);
        let result = parse_archive(&gz_data, 0);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let info = result.unwrap();
        assert_eq!(info.format, ArchiveFormat::TarGzip);
    }

    #[test]
    fn test_zip_bomb_detection() {
        // Create a ZIP where one entry claims a huge uncompressed size.
        // We cannot easily create a true zip bomb in tests, but we can
        // test the ratio detection logic with a highly compressible payload.
        let zeros = vec![0u8; 1024];
        let data = make_zip(&[("zeros.bin", &zeros)]);
        // With Stored compression, ratio ~ 1:1, should pass.
        let result = parse_archive(&data, 0);
        assert!(result.is_ok());
        let info = result.unwrap();
        assert!(!info.is_zip_bomb);
    }

    #[test]
    fn test_max_recursion_depth() {
        let result = parse_archive(&[], 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_zip() {
        let inner_zip = make_zip(&[("inner.txt", b"nested content")]);
        let outer_zip = make_zip(&[("inner.zip", &inner_zip)]);
        let result = parse_archive(&outer_zip, 0);
        assert!(result.is_ok());
        let info = result.unwrap();
        assert!(
            info.entries.iter().any(|e| e.is_nested_archive),
            "should detect nested archive"
        );
    }

    #[test]
    fn test_extract_features_zip() {
        let data = make_zip(&[("test.txt", b"content")]);
        let features = ArchiveParser::extract_features(&data);
        assert!(features.is_ok());
        let fv = features.unwrap();
        assert!(!fv.numeric.is_empty());
        // entry_count feature (index 4) should be 1.0.
        assert_eq!(fv.numeric[4], 1.0);
    }

    #[test]
    fn test_detect_format_zip() {
        let data = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00];
        assert_eq!(detect_archive_format(&data).unwrap(), ArchiveFormat::Zip);
    }

    #[test]
    fn test_detect_format_gzip() {
        let data = [0x1F, 0x8B, 0x08, 0x00];
        assert_eq!(detect_archive_format(&data).unwrap(), ArchiveFormat::Gzip);
    }

    #[test]
    fn test_detect_format_unknown() {
        let data = [0x00, 0x01, 0x02, 0x03];
        assert!(detect_archive_format(&data).is_err());
    }
}
