use std::io::Cursor;

use base64::Engine;
use image::{ImageEncoder, RgbImage};
use ndarray::{s, Array3};
use rayon::prelude::*;

use super::types::SliceResult;

/// Normalize a u16 slice to u8 [0, 255].
fn normalize_u16_to_u8(slice: &ndarray::ArrayView2<u16>) -> Vec<u8> {
    let max_val = *slice.iter().max().unwrap_or(&1);
    if max_val == 0 {
        return vec![0u8; slice.len()];
    }
    slice
        .iter()
        .map(|&v| ((v as f64 / max_val as f64) * 255.0).min(255.0) as u8)
        .collect()
}

/// Encode grayscale pixels as a base64 PNG string.
fn encode_gray_png_base64(pixels: &[u8], width: u32, height: u32) -> Result<String, String> {
    let img = image::GrayImage::from_raw(width, height, pixels.to_vec())
        .ok_or("Failed to create grayscale image")?;
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buf);
    encoder
        .write_image(img.as_raw(), width, height, image::ColorType::L8.into())
        .map_err(|e| format!("PNG encode error: {}", e))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

/// Encode RGB pixels as a base64 PNG string.
fn encode_rgb_png_base64(pixels: &[u8], width: u32, height: u32) -> Result<String, String> {
    let img =
        RgbImage::from_raw(width, height, pixels.to_vec()).ok_or("Failed to create RGB image")?;
    let mut buf: Vec<u8> = Vec::new();
    let cursor = Cursor::new(&mut buf);
    let encoder = image::codecs::png::PngEncoder::new(cursor);
    encoder
        .write_image(img.as_raw(), width, height, image::ColorType::Rgb8.into())
        .map_err(|e| format!("PNG encode error: {}", e))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

/// Pre-compute the entire 3D border volume from the annotation volume.
/// Parallelized across slices with rayon.
pub fn precompute_border_volume_parallel(annotation: &Array3<u32>) -> Array3<u8> {
    let (depth, height, width) = (
        annotation.shape()[0],
        annotation.shape()[1],
        annotation.shape()[2],
    );

    // Compute each slice independently in parallel
    let slices: Vec<Vec<u8>> = (0..depth)
        .into_par_iter()
        .map(|z| {
            let ann_slice = annotation.slice(s![z, .., ..]);
            let mut edges = vec![0u8; height * width];
            for y in 0..height {
                for x in 0..width {
                    let val = ann_slice[[y, x]];
                    if x + 1 < width && ann_slice[[y, x + 1]] != val {
                        edges[y * width + x] = 255;
                        edges[y * width + x + 1] = 255;
                    }
                    if y + 1 < height && ann_slice[[y + 1, x]] != val {
                        edges[y * width + x] = 255;
                        edges[(y + 1) * width + x] = 255;
                    }
                }
            }
            edges
        })
        .collect();

    let flat: Vec<u8> = slices.into_iter().flatten().collect();
    Array3::from_shape_vec((depth, height, width), flat).expect("Border volume reshape failed")
}

/// Fast path: return raw border bytes for a single coronal slice from the
/// pre-computed border volume. No PNG encoding — just base64 of raw u8 pixels.
pub fn get_precomputed_border_slice(
    border_volume: &Array3<u8>,
    ap_index: usize,
) -> (String, u32, u32) {
    let slice = border_volume.slice(s![ap_index, .., ..]);
    let height = slice.shape()[0] as u32;
    let width = slice.shape()[1] as u32;

    // Contiguous raw bytes
    let raw: Vec<u8> = slice.iter().copied().collect();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
    (b64, width, height)
}

/// Extract region boundaries from an annotation slice.
/// Same algorithm as Python `_annotation_to_boundaries`.
fn compute_borders(annotation_slice: &ndarray::ArrayView2<u32>) -> Vec<u8> {
    let (h, w) = (annotation_slice.shape()[0], annotation_slice.shape()[1]);
    let mut edges = vec![0u8; h * w];

    for y in 0..h {
        for x in 0..w {
            let val = annotation_slice[[y, x]];
            // Horizontal edge
            if x + 1 < w && annotation_slice[[y, x + 1]] != val {
                edges[y * w + x] = 255;
                edges[y * w + x + 1] = 255;
            }
            // Vertical edge
            if y + 1 < h && annotation_slice[[y + 1, x]] != val {
                edges[y * w + x] = 255;
                edges[(y + 1) * w + x] = 255;
            }
        }
    }

    edges
}

/// Blend reference (grayscale) with borders (green #00ff64 at 40% opacity).
fn composite_reference_borders(
    reference: &[u8],
    borders: &[u8],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let n_pixels = (width * height) as usize;
    let mut rgb = vec![0u8; n_pixels * 3];
    let opacity = 0.4f64;

    for i in 0..n_pixels {
        let ref_val = reference[i] as f64;
        let is_border = borders[i] > 0;

        if is_border {
            // Green #00ff64 blended at 40% over the reference grayscale
            rgb[i * 3] = ((1.0 - opacity) * ref_val + opacity * 0.0) as u8; // R
            rgb[i * 3 + 1] = ((1.0 - opacity) * ref_val + opacity * 255.0) as u8; // G
            rgb[i * 3 + 2] = ((1.0 - opacity) * ref_val + opacity * 100.0) as u8;
        // B
        } else {
            rgb[i * 3] = reference[i];
            rgb[i * 3 + 1] = reference[i];
            rgb[i * 3 + 2] = reference[i];
        }
    }

    rgb
}

/// Fast path: extract only the border pixels as base64 PNG (no reference, no composite).
pub fn extract_borders_only(
    annotation_volume: &Array3<u32>,
    ap_index: usize,
) -> Result<(String, u32, u32), String> {
    let ann_slice = annotation_volume.slice(s![ap_index, .., ..]);
    let height = ann_slice.shape()[0] as u32;
    let width = ann_slice.shape()[1] as u32;
    let border_pixels = compute_borders(&ann_slice);
    let b64 = encode_gray_png_base64(&border_pixels, width, height)?;
    Ok((b64, width, height))
}

/// Extract a coronal slice at a given AP index, returning reference, borders, and composite.
pub fn extract_coronal_slice(
    reference_volume: &Array3<u16>,
    annotation_volume: &Array3<u32>,
    ap_index: usize,
) -> Result<SliceResult, String> {
    let ref_slice = reference_volume.slice(s![ap_index, .., ..]);
    let ann_slice = annotation_volume.slice(s![ap_index, .., ..]);

    let height = ref_slice.shape()[0] as u32;
    let width = ref_slice.shape()[1] as u32;

    // Reference: normalize to u8 grayscale
    let ref_pixels = normalize_u16_to_u8(&ref_slice);
    let reference_b64 = encode_gray_png_base64(&ref_pixels, width, height)?;

    // Borders: edge detection on annotation
    let border_pixels = compute_borders(&ann_slice);
    let borders_b64 = encode_gray_png_base64(&border_pixels, width, height)?;

    // Composite: reference + green borders
    let composite_pixels = composite_reference_borders(&ref_pixels, &border_pixels, width, height);
    let composite_b64 = encode_rgb_png_base64(&composite_pixels, width, height)?;

    Ok(SliceResult {
        reference: reference_b64,
        borders: borders_b64,
        composite: composite_b64,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::loader::{find_atlas_dir, load_tiff_u16, load_tiff_u32};

    #[test]
    fn test_extract_coronal_slice() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let reference = load_tiff_u16(&dir.join("reference.tiff")).unwrap();
        let annotation = load_tiff_u32(&dir.join("annotation.tiff")).unwrap();

        // Slice at roughly the middle of the volume
        let mid = reference.shape()[0] / 2;
        let result = extract_coronal_slice(&reference, &annotation, mid).unwrap();

        assert!(!result.reference.is_empty());
        assert!(!result.borders.is_empty());
        assert!(!result.composite.is_empty());
        assert!(result.width > 0);
        assert!(result.height > 0);

        println!(
            "Slice at index {}: {}x{}, ref_b64_len={}, borders_b64_len={}, composite_b64_len={}",
            mid,
            result.width,
            result.height,
            result.reference.len(),
            result.borders.len(),
            result.composite.len(),
        );
    }
}
