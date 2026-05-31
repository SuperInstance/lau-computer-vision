//! Feature detection — Harris corners, blob detection basics.

use crate::image::GrayImage;
use crate::filter;

/// A detected feature (corner, blob, etc.).
#[derive(Debug, Clone)]
pub struct FeaturePoint {
    pub x: usize,
    pub y: usize,
    pub response: f64,
    pub scale: f64,
}

/// Harris corner response for every pixel.
pub fn harris_response(img: &GrayImage, k: f64) -> GrayImage {
    // Compute image gradients via Sobel
    let (_, _, gx, gy) = filter::sobel(img);

    let w = img.width();
    let h = img.height();

    // Compute products
    let mut ixx = GrayImage::new(w, h);
    let mut ixy = GrayImage::new(w, h);
    let mut iyy = GrayImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let dx = gx.data[y * w + x];
            let dy = gy.data[y * w + x];
            ixx.data[y * w + x] = dx * dx;
            ixy.data[y * w + x] = dx * dy;
            iyy.data[y * w + x] = dy * dy;
        }
    }

    // Smooth the products (window function)
    let gauss_kernel = filter::gaussian_kernel(1.5, 7);
    let sxx = ixx.convolve(&gauss_kernel).unwrap_or(ixx);
    let sxy = ixy.convolve(&gauss_kernel).unwrap_or(ixy);
    let syy = iyy.convolve(&gauss_kernel).unwrap_or(iyy);

    // Harris response R = det(M) - k * trace(M)^2
    let mut response = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let a = sxx.data[y * w + x];
            let b = sxy.data[y * w + x];
            let c = syy.data[y * w + x];
            let det = a * c - b * b;
            let trace = a + c;
            response.data[y * w + x] = det - k * trace * trace;
        }
    }
    response
}

/// Detect Harris corners, returning feature points sorted by response (descending).
pub fn harris_corners(img: &GrayImage, k: f64, threshold: f64, nms_radius: usize) -> Vec<FeaturePoint> {
    let response = harris_response(img, k);
    let w = img.width();
    let h = img.height();

    let mut points: Vec<FeaturePoint> = Vec::new();

    for y in nms_radius..h.saturating_sub(nms_radius) {
        for x in nms_radius..w.saturating_sub(nms_radius) {
            let r = response.data[y * w + x];
            if r <= threshold { continue; }

            // Non-maximum suppression
            let mut is_max = true;
            'outer: for dy in -(nms_radius as isize)..=(nms_radius as isize) {
                for dx in -(nms_radius as isize)..=(nms_radius as isize) {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    if response.data[ny * w + nx] > r {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                points.push(FeaturePoint { x, y, response: r, scale: 1.0 });
            }
        }
    }

    points.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    points
}

/// Simple blob detection using Laplacian of Gaussian (LoG) at multiple scales.
pub fn detect_blobs(img: &GrayImage, min_sigma: f64, max_sigma: f64, num_scales: usize, threshold: f64) -> Vec<FeaturePoint> {
    let mut blobs: Vec<FeaturePoint> = Vec::new();

    for i in 0..num_scales {
        let sigma = min_sigma + (max_sigma - min_sigma) * i as f64 / (num_scales - 1).max(1) as f64;
        let blurred = filter::gaussian_blur(img, sigma);
        let log_resp = filter::laplacian(&blurred);

        // Scale-normalized response: sigma^2 * |LoG|
        let scale_factor = sigma * sigma;

        for y in 1..img.height().saturating_sub(1) {
            for x in 1..img.width().saturating_sub(1) {
                let r = log_resp.data[y * img.width() + x].abs() * scale_factor;
                if r > threshold {
                    blobs.push(FeaturePoint {
                        x, y, response: r, scale: sigma,
                    });
                }
            }
        }
    }

    blobs.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    blobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harris_corner_detection() {
        // Create a simple checkerboard-like pattern with corners
        let mut img = GrayImage::new(20, 20);
        // White square on dark background — 4 corners
        for y in 5..15 {
            for x in 5..15 {
                img.set(x, y, 200.0).unwrap();
            }
        }

        let corners = harris_corners(&img, 0.04, 1.0, 3);
        // Should detect corners at approximately (5,5), (14,5), (5,14), (14,14)
        assert!(corners.len() >= 4, "Expected at least 4 corners, got {}", corners.len());
    }

    #[test]
    fn test_harris_response_uniform_is_zero() {
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.set(x, y, 128.0).unwrap();
            }
        }
        let resp = harris_response(&img, 0.04);
        // Only check interior (boundaries have artifacts from zero-padding)
        for y in 4..16 {
            for x in 4..16 {
                let r = resp.data[y * 20 + x];
                assert!(r.abs() < 1.0, "Interior should have ~0 response at ({},{}), got {}", x, y, r);
            }
        }
    }

    #[test]
    fn test_harris_response_edge_negative() {
        // Pure vertical edge: strong gradient in x, none in y
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..10 {
                img.set(x, y, 0.0).unwrap();
            }
            for x in 10..20 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let resp = harris_response(&img, 0.04);
        // Edge points should have negative (or near-zero) response
        let r = resp.get(10, 10).unwrap();
        assert!(r <= 1.0, "Edge should have low/negative Harris response, got {}", r);
    }

    #[test]
    fn test_blob_detection_simple() {
        // Bright Gaussian-like blob in center
        let mut img = GrayImage::new(21, 21);
        for y in 0..21 {
            for x in 0..21 {
                let dx = x as f64 - 10.0;
                let dy = y as f64 - 10.0;
                let val = 255.0 * (-(dx * dx + dy * dy) / 20.0).exp();
                img.set(x, y, val).unwrap();
            }
        }

        let blobs = detect_blobs(&img, 1.0, 5.0, 5, 10.0);
        assert!(!blobs.is_empty(), "Should detect at least one blob");
        // Closest blob to center
        let closest = blobs.iter().min_by_key(|b| {
            ((b.x as f64 - 10.0).powi(2) + (b.y as f64 - 10.0).powi(2)) as u64
        }).unwrap();
        assert!((closest.x as f64 - 10.0).abs() < 3.0);
        assert!((closest.y as f64 - 10.0).abs() < 3.0);
    }

    #[test]
    fn test_harris_corners_empty_image() {
        let img = GrayImage::new(10, 10);
        let corners = harris_corners(&img, 0.04, 1.0, 2);
        assert!(corners.is_empty());
    }

    #[test]
    fn test_feature_point_fields() {
        let fp = FeaturePoint { x: 5, y: 10, response: 42.0, scale: 2.0 };
        assert_eq!(fp.x, 5);
        assert_eq!(fp.y, 10);
        assert_eq!(fp.response, 42.0);
    }
}
