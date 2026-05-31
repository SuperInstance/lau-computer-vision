//! Optical flow basics — Lucas-Kanade.

use crate::image::GrayImage;
use crate::filter;

/// A 2D flow vector at a pixel.
#[derive(Debug, Clone, Copy)]
pub struct FlowVector {
    pub u: f64,
    pub v: f64,
}

/// Compute Lucas-Kanade optical flow between two frames.
/// Returns flow at every pixel (zero where flow cannot be computed).
pub fn lucas_kanade(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> GrayImage {
    let w = prev.width();
    let h = prev.height();

    // Temporal gradient
    let mut it = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            it.data[y * w + x] = curr.data[y * w + x] - prev.data[y * w + x];
        }
    }

    // Spatial gradients
    let (_, _, gx, gy) = filter::sobel(prev);

    let half = window_size / 2;

    // We'll encode u and v in separate images
    let mut flow_u = GrayImage::new(w, h);
    let mut flow_v = GrayImage::new(w, h);

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut sum_ix2 = 0.0f64;
            let mut sum_iy2 = 0.0f64;
            let mut sum_ixiy = 0.0f64;
            let mut sum_ixit = 0.0f64;
            let mut sum_iyit = 0.0f64;

            for dy in -(half as isize)..=(half as isize) {
                for dx in -(half as isize)..=(half as isize) {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    let ix = gx.data[ny * w + nx];
                    let iy = gy.data[ny * w + nx];
                    let it_val = it.data[ny * w + nx];

                    sum_ix2 += ix * ix;
                    sum_iy2 += iy * iy;
                    sum_ixiy += ix * iy;
                    sum_ixit += ix * it_val;
                    sum_iyit += iy * it_val;
                }
            }

            // Solve 2x2 system: [ix2, ixiy; ixiy, iy2] * [u; v] = [-ixit; -iyit]
            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if det.abs() > 1e-6 {
                let u = (sum_iy2 * (-sum_ixit) - sum_ixiy * (-sum_iyit)) / det;
                let v = (sum_ix2 * (-sum_iyit) - sum_ixiy * (-sum_ixit)) / det;
                flow_u.data[y * w + x] = u;
                flow_v.data[y * w + x] = v;
            }
        }
    }

    // Return flow magnitude image
    flow_u.zip_with(&flow_v, |u, v| (u * u + v * v).sqrt()).unwrap()
}

/// Compute Lucas-Kanade optical flow returning both u and v components.
pub fn lucas_kanade_uv(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> (GrayImage, GrayImage) {
    let w = prev.width();
    let h = prev.height();

    let mut it = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            it.data[y * w + x] = curr.data[y * w + x] - prev.data[y * w + x];
        }
    }

    let (_, _, gx, gy) = filter::sobel(prev);

    let half = window_size / 2;
    let mut flow_u = GrayImage::new(w, h);
    let mut flow_v = GrayImage::new(w, h);

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut sum_ix2 = 0.0f64;
            let mut sum_iy2 = 0.0f64;
            let mut sum_ixiy = 0.0f64;
            let mut sum_ixit = 0.0f64;
            let mut sum_iyit = 0.0f64;

            for dy in -(half as isize)..=(half as isize) {
                for dx in -(half as isize)..=(half as isize) {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    let ix = gx.data[ny * w + nx];
                    let iy = gy.data[ny * w + nx];
                    let it_val = it.data[ny * w + nx];

                    sum_ix2 += ix * ix;
                    sum_iy2 += iy * iy;
                    sum_ixiy += ix * iy;
                    sum_ixit += ix * it_val;
                    sum_iyit += iy * it_val;
                }
            }

            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if det.abs() > 1e-6 {
                flow_u.data[y * w + x] = (sum_iy2 * (-sum_ixit) - sum_ixiy * (-sum_iyit)) / det;
                flow_v.data[y * w + x] = (sum_ix2 * (-sum_iyit) - sum_ixiy * (-sum_ixit)) / det;
            }
        }
    }

    (flow_u, flow_v)
}

/// Compute dense optical flow between two frames and return flow vectors.
pub fn dense_flow(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> Vec<Vec<Option<FlowVector>>> {
    let w = prev.width();
    let h = prev.height();

    let (u_img, v_img) = lucas_kanade_uv(prev, curr, window_size);

    let mut result = vec![vec![None; w]; h];
    for y in 0..h {
        for x in 0..w {
            let u = u_img.data[y * w + x];
            let v = v_img.data[y * w + x];
            if u != 0.0 || v != 0.0 {
                result[y][x] = Some(FlowVector { u, v });
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate_image(img: &GrayImage, shift_x: usize) -> GrayImage {
        let w = img.width();
        let h = img.height();
        let mut shifted = GrayImage::new(w, h);
        for y in 0..h {
            for x in shift_x..w {
                shifted.data[y * w + x] = img.data[y * w + (x - shift_x)];
            }
        }
        shifted
    }

    #[test]
    fn test_lk_static_image_zero_flow() {
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.set(x, y, ((x * 7 + y * 13) % 256) as f64).unwrap();
            }
        }
        let flow_mag = lucas_kanade(&img, &img, 5);
        // All flow should be zero (same image)
        let max_flow = flow_mag.data().iter().cloned().fold(0.0f64, f64::max);
        assert!(max_flow < 1.0, "Static image should have ~0 flow, max = {}", max_flow);
    }

    #[test]
    fn test_lk_detects_horizontal_motion() {
        // Create 2D texture pattern (checkerboard-like) so both gradients are non-zero
        let mut img = GrayImage::new(40, 30);
        for y in 0..30 {
            for x in 0..40 {
                let val = (((x as f64 * 0.3).sin() + (y as f64 * 0.3).sin()) * 60.0 + 128.0).max(0.0).min(255.0);
                img.set(x, y, val).unwrap();
            }
        }
        let shifted = translate_image(&img, 1);
        let (u_img, _v_img) = lucas_kanade_uv(&img, &shifted, 7);

        // Check interior pixels for non-zero horizontal flow
        let mut count_nonzero = 0;
        for y in 6..24 {
            for x in 8..32 {
                let u = u_img.data[y * 40 + x];
                if u.abs() > 0.01 {
                    count_nonzero += 1;
                }
            }
        }
        assert!(count_nonzero > 5, "Should have flow vectors, got {}", count_nonzero);
    }

    #[test]
    fn test_lk_flow_magnitude_nonzero() {
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.set(x, y, (x * 12 + y * 7) as f64).unwrap();
            }
        }
        let shifted = translate_image(&img, 2);
        let mag = lucas_kanade(&img, &shifted, 5);
        let max_mag = mag.data().iter().cloned().fold(0.0f64, f64::max);
        assert!(max_mag > 0.5, "Flow magnitude should be nonzero, max = {}", max_mag);
    }

    #[test]
    fn test_dense_flow_structure() {
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.set(x, y, (x * 25) as f64).unwrap();
            }
        }
        let shifted = translate_image(&img, 1);
        let flow = dense_flow(&img, &shifted, 3);
        assert_eq!(flow.len(), 10);
        assert_eq!(flow[0].len(), 10);
    }

    #[test]
    fn test_flow_vector_fields() {
        let fv = FlowVector { u: 1.5, v: -0.5 };
        assert_eq!(fv.u, 1.5);
        assert_eq!(fv.v, -0.5);
    }

    #[test]
    fn test_lk_uniform_image_no_flow() {
        let img = GrayImage::from_vec(10, 10, vec![128.0; 100]).unwrap();
        let shifted = img.clone();
        let mag = lucas_kanade(&img, &shifted, 3);
        let max_mag = mag.data().iter().cloned().fold(0.0f64, f64::max);
        assert!(max_mag < 1.0, "Uniform image should have no flow");
    }
}
