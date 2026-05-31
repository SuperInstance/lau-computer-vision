//! Convolution and filtering — blur, sharpen, edge detection (Sobel, Canny basics).

use crate::image::GrayImage;

/// Box blur kernel of given size.
pub fn box_blur_kernel(size: usize) -> Vec<Vec<f64>> {
    let val = 1.0 / (size * size) as f64;
    vec![vec![val; size]; size]
}

/// Gaussian blur kernel (approximate, separable).
pub fn gaussian_kernel(sigma: f64, size: usize) -> Vec<Vec<f64>> {
    let center = size as f64 / 2.0;
    let mut kernel = vec![vec![0.0; size]; size];
    let mut sum = 0.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let val = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
            kernel[y][x] = val;
            sum += val;
        }
    }
    // Normalize
    for row in &mut kernel {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    kernel
}

/// Apply Gaussian blur.
pub fn gaussian_blur(img: &GrayImage, sigma: f64) -> GrayImage {
    let size = ((sigma * 6.0).ceil() as usize).max(3) | 1; // ensure odd
    let kernel = gaussian_kernel(sigma, size);
    img.convolve(&kernel).unwrap_or_else(|_| img.clone())
}

/// Apply box blur.
pub fn box_blur(img: &GrayImage, size: usize) -> GrayImage {
    let kernel = box_blur_kernel(size);
    img.convolve(&kernel).unwrap_or_else(|_| img.clone())
}

/// Sharpening kernel (unsharp mask style).
pub fn sharpen_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![ 0.0, -1.0,  0.0],
        vec![-1.0,  5.0, -1.0],
        vec![ 0.0, -1.0,  0.0],
    ]
}

/// Sharpen image.
pub fn sharpen(img: &GrayImage) -> GrayImage {
    let kernel = sharpen_kernel();
    img.convolve(&kernel).unwrap_or_else(|_| img.clone())
}

/// Sobel Gx kernel.
pub fn sobel_gx() -> Vec<Vec<f64>> {
    vec![
        vec![-1.0, 0.0, 1.0],
        vec![-2.0, 0.0, 2.0],
        vec![-1.0, 0.0, 1.0],
    ]
}

/// Sobel Gy kernel.
pub fn sobel_gy() -> Vec<Vec<f64>> {
    vec![
        vec![-1.0, -2.0, -1.0],
        vec![ 0.0,  0.0,  0.0],
        vec![ 1.0,  2.0,  1.0],
    ]
}

/// Compute Sobel gradients, returning (magnitude, direction, gx, gy).
pub fn sobel(img: &GrayImage) -> (GrayImage, GrayImage, GrayImage, GrayImage) {
    let gx_img = img.convolve(&sobel_gx()).unwrap_or_else(|_| img.clone());
    let gy_img = img.convolve(&sobel_gy()).unwrap_or_else(|_| img.clone());

    let w = img.width();
    let h = img.height();
    let mut mag = GrayImage::new(w, h);
    let mut dir = GrayImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let gx = gx_img.data[y * w + x];
            let gy = gy_img.data[y * w + x];
            mag.data[y * w + x] = (gx * gx + gy * gy).sqrt();
            dir.data[y * w + x] = gy.atan2(gx);
        }
    }
    (mag, dir, gx_img, gy_img)
}

/// Simple Canny edge detection.
pub fn canny(img: &GrayImage, low: f64, high: f64) -> GrayImage {
    // 1. Gaussian blur
    let blurred = gaussian_blur(img, 1.4);

    // 2. Sobel gradients
    let (mag, dir, _, _) = sobel(&blurred);

    let w = img.width();
    let h = img.height();

    // 3. Non-maximum suppression
    let mut nms = GrayImage::new(w, h);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let angle = dir.data[y * w + x].to_degrees().rem_euclid(180.0);
            let m = mag.data[y * w + x];

            let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
                (mag.data[y * w + x + 1], mag.data[y * w + x - 1])
            } else if angle < 67.5 {
                (mag.data[(y - 1) * w + x + 1], mag.data[(y + 1) * w + x - 1])
            } else if angle < 112.5 {
                (mag.data[(y - 1) * w + x], mag.data[(y + 1) * w + x])
            } else {
                (mag.data[(y - 1) * w + x - 1], mag.data[(y + 1) * w + x + 1])
            };

            nms.data[y * w + x] = if m >= n1 && m >= n2 { m } else { 0.0 };
        }
    }

    // 4. Double threshold + hysteresis
    let mut edges = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = nms.data[y * w + x];
            if v >= high {
                edges.data[y * w + x] = 255.0;
            } else if v >= low {
                edges.data[y * w + x] = 128.0; // weak
            }
        }
    }

    // Hysteresis: promote weak pixels connected to strong
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if edges.data[y * w + x] == 128.0 {
                    // Check 8-connected neighbors
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            if edges.data[ny * w + nx] == 255.0 {
                                edges.data[y * w + x] = 255.0;
                                changed = true;
                                break;
                            }
                        }
                        if edges.data[y * w + x] == 255.0 { break; }
                    }
                }
            }
        }
    }

    // Remove remaining weak
    for v in edges.data_mut().iter_mut() {
        if *v < 255.0 { *v = 0.0; }
    }

    edges
}

/// Laplacian kernel.
pub fn laplacian_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![0.0,  1.0, 0.0],
        vec![1.0, -4.0, 1.0],
        vec![0.0,  1.0, 0.0],
    ]
}

/// Apply Laplacian filter.
pub fn laplacian(img: &GrayImage) -> GrayImage {
    img.convolve(&laplacian_kernel()).unwrap_or_else(|_| img.clone())
}

/// Prewitt Gx kernel.
pub fn prewitt_gx() -> Vec<Vec<f64>> {
    vec![
        vec![-1.0, 0.0, 1.0],
        vec![-1.0, 0.0, 1.0],
        vec![-1.0, 0.0, 1.0],
    ]
}

/// Prewitt Gy kernel.
pub fn prewitt_gy() -> Vec<Vec<f64>> {
    vec![
        vec![-1.0, -1.0, -1.0],
        vec![ 0.0,  0.0,  0.0],
        vec![ 1.0,  1.0,  1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_blur_kernel_sums_to_one() {
        let k = box_blur_kernel(3);
        let sum: f64 = k.iter().flat_map(|r| r.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel(1.0, 5);
        let sum: f64 = k.iter().flat_map(|r| r.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_blur_preserves_brightness() {
        let mut img = GrayImage::new(30, 30);
        for y in 0..30 {
            for x in 0..30 {
                img.set(x, y, 128.0).unwrap();
            }
        }
        let blurred = gaussian_blur(&img, 1.0);
        // Check interior away from zero-padded boundary
        let mut sum = 0.0;
        let mut count = 0;
        for y in 5..25 {
            for x in 5..25 {
                sum += blurred.get(x, y).unwrap();
                count += 1;
            }
        }
        let avg = sum / count as f64;
        assert!((avg - 128.0).abs() < 5.0, "avg = {}", avg);
    }

    #[test]
    fn test_box_blur_smooths() {
        let mut img = GrayImage::new(11, 11);
        img.set(5, 5, 255.0).unwrap();
        let blurred = box_blur(&img, 3);
        // Center should be less than original
        assert!(blurred.get(5, 5).unwrap() < 255.0);
        // Neighbors should have some value
        assert!(blurred.get(4, 5).unwrap() > 0.0);
    }

    #[test]
    fn test_sobel_horizontal_edge() {
        // Top half white, bottom half black
        let mut img = GrayImage::new(10, 10);
        for y in 0..5 {
            for x in 0..10 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let (mag, _, _, _) = sobel(&img);
        // Strong response at the boundary
        assert!(mag.get(5, 4).unwrap() > 100.0);
        // No response well away from boundary
        assert!(mag.get(5, 2).unwrap() < 50.0, "far from edge should have low response");
    }

    #[test]
    fn test_sharpen_increases_contrast() {
        let mut img = GrayImage::new(9, 9);
        img.set(4, 4, 200.0).unwrap();
        img.set(4, 3, 50.0).unwrap();
        let sharp = sharpen(&img);
        // Center should be amplified
        assert!(sharp.get(4, 4).unwrap() > 200.0);
    }

    #[test]
    fn test_sobel_kernels() {
        let gx = sobel_gx();
        let gy = sobel_gy();
        assert_eq!(gx[0][0], -1.0);
        assert_eq!(gx[1][1], 0.0);
        assert_eq!(gy[0][0], -1.0);
        assert_eq!(gy[1][1], 0.0);
    }

    #[test]
    fn test_canny_detects_edge() {
        let mut img = GrayImage::new(20, 20);
        for y in 0..10 {
            for x in 0..20 {
                img.set(x, y, 200.0).unwrap();
            }
        }
        let edges = canny(&img, 20.0, 60.0);
        // There should be edge pixels near y=10
        let mut has_edge = false;
        for x in 0..20 {
            for dy in 8..13 {
                if edges.get(x, dy).unwrap() > 0.0 {
                    has_edge = true;
                }
            }
        }
        assert!(has_edge, "Canny should detect horizontal edge");
    }

    #[test]
    fn test_laplacian_kernel() {
        let k = laplacian_kernel();
        // Sum should be 0 (edge detector)
        let sum: f64 = k.iter().flat_map(|r| r.iter()).sum();
        assert!((sum).abs() < 1e-10);
    }

    #[test]
    fn test_prewitt_kernels() {
        let gx = prewitt_gx();
        let gy = prewitt_gy();
        assert_eq!(gx.len(), 3);
        assert_eq!(gy.len(), 3);
        assert_eq!(gx[0][2], 1.0);
        assert_eq!(gy[2][0], 1.0);
    }
}
