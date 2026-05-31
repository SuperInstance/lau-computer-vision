//! Image transforms — Hough transform for lines, affine transforms.

use crate::image::GrayImage;

/// A detected line in Hough space (rho, theta).
#[derive(Debug, Clone)]
pub struct HoughLine {
    pub rho: f64,
    pub theta: f64,
    pub votes: usize,
}

/// Hough transform accumulator for line detection.
pub struct HoughLineDetector {
    pub rho_max: f64,
    pub theta_bins: usize,
    pub rho_bins: usize,
    accumulator: Vec<Vec<usize>>,
    theta_step: f64,
    rho_step: f64,
}

impl HoughLineDetector {
    /// Create a new Hough line detector.
    pub fn new(width: usize, height: usize, theta_bins: usize, rho_bins: usize) -> Self {
        let rho_max = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        let theta_step = std::f64::consts::PI / theta_bins as f64;
        let rho_step = 2.0 * rho_max / rho_bins as f64;
        Self {
            rho_max,
            theta_bins,
            rho_bins,
            accumulator: vec![vec![0; rho_bins]; theta_bins],
            theta_step,
            rho_step,
        }
    }

    /// Vote for all lines passing through each edge pixel.
    pub fn detect(&mut self, edge_image: &GrayImage) {
        let rho_offset = self.rho_max;
        for y in 0..edge_image.height() {
            for x in 0..edge_image.width() {
                if edge_image.data[y * edge_image.width() + x] > 0.0 {
                    let xf = x as f64;
                    let yf = y as f64;
                    for t in 0..self.theta_bins {
                        let theta = t as f64 * self.theta_step;
                        let rho = xf * theta.cos() + yf * theta.sin();
                        let r_bin = ((rho + rho_offset) / self.rho_step) as isize;
                        if r_bin >= 0 && (r_bin as usize) < self.rho_bins {
                            self.accumulator[t][r_bin as usize] += 1;
                        }
                    }
                }
            }
        }
    }

    /// Get the accumulator value at (theta_bin, rho_bin).
    pub fn get(&self, theta_bin: usize, rho_bin: usize) -> usize {
        self.accumulator[theta_bin][rho_bin]
    }

    /// Extract peaks from the accumulator above the given threshold.
    pub fn peaks(&self, threshold: usize, nms_radius: usize) -> Vec<HoughLine> {
        let mut lines = Vec::new();

        for t in 0..self.theta_bins {
            for r in 0..self.rho_bins {
                let votes = self.accumulator[t][r];
                if votes < threshold { continue; }

                // NMS
                let mut is_max = true;
                let t_start = t.saturating_sub(nms_radius);
                let t_end = (t + nms_radius + 1).min(self.theta_bins);
                let r_start = r.saturating_sub(nms_radius);
                let r_end = (r + nms_radius + 1).min(self.rho_bins);

                'outer: for tt in t_start..t_end {
                    for rr in r_start..r_end {
                        if tt == t && rr == r { continue; }
                        if self.accumulator[tt][rr] > votes {
                            is_max = false;
                            break 'outer;
                        }
                    }
                }

                if is_max {
                    let theta = t as f64 * self.theta_step;
                    let rho = r as f64 * self.rho_step - self.rho_max;
                    lines.push(HoughLine { rho, theta, votes });
                }
            }
        }

        lines.sort_by(|a, b| b.votes.cmp(&a.votes));
        lines
    }
}

/// Apply affine transformation using a 2x3 matrix [[a, b, tx], [c, d, ty]].
/// Uses inverse mapping with bilinear interpolation.
pub fn affine_transform(img: &GrayImage, matrix: &[[f64; 3]; 2]) -> GrayImage {
    let w = img.width();
    let h = img.height();
    let mut out = GrayImage::new(w, h);

    // Inverse of the 2x2 part
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];
    let det = a * d - b * c;

    if det.abs() < 1e-10 {
        return out; // Degenerate
    }

    let inv_det = 1.0 / det;
    let inv_a = d * inv_det;
    let inv_b = -b * inv_det;
    let inv_c = -c * inv_det;
    let inv_d = a * inv_det;

    let tx = matrix[0][2];
    let ty = matrix[1][2];

    for y in 0..h {
        for x in 0..w {
            // Inverse map
            let sx = inv_a * (x as f64 - tx) + inv_b * (y as f64 - ty);
            let sy = inv_c * (x as f64 - tx) + inv_d * (y as f64 - ty);

            // Bilinear interpolation
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;

            let v00 = img.get_padded(x0, y0);
            let v10 = img.get_padded(x1, y0);
            let v01 = img.get_padded(x0, y1);
            let v11 = img.get_padded(x1, y1);

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

            out.data_mut()[y * w + x] = val;
        }
    }
    out
}

/// Create a rotation matrix for affine_transform.
pub fn rotation_matrix(angle_rad: f64, cx: f64, cy: f64) -> [[f64; 3]; 2] {
    let cos = angle_rad.cos();
    let sin = angle_rad.sin();
    [
        [cos, -sin, -cx * cos + cy * sin + cx],
        [sin, cos, -cx * sin - cy * cos + cy],
    ]
}

/// Create a translation matrix.
pub fn translation_matrix(tx: f64, ty: f64) -> [[f64; 3]; 2] {
    [
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
    ]
}

/// Create a scaling matrix.
pub fn scale_matrix(sx: f64, sy: f64, cx: f64, cy: f64) -> [[f64; 3]; 2] {
    [
        [sx, 0.0, cx * (1.0 - sx)],
        [0.0, sy, cy * (1.0 - sy)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge_image() -> GrayImage {
        let mut img = GrayImage::new(50, 50);
        // Horizontal line at y=25
        for x in 0..50 {
            img.set(x, 25, 255.0).unwrap();
        }
        img
    }

    #[test]
    fn test_hough_horizontal_line() {
        let edge = make_edge_image();
        let mut detector = HoughLineDetector::new(50, 50, 180, 100);
        detector.detect(&edge);
        let peaks = detector.peaks(10, 5);
        assert!(!peaks.is_empty(), "Should detect at least one line");

        // The horizontal line at y=25 should correspond to theta≈π/2, rho≈25
        let found = peaks.iter().any(|l| {
            let theta_ok = (l.theta - std::f64::consts::FRAC_PI_2).abs() < 0.2;
            let rho_ok = (l.rho - 25.0).abs() < 5.0;
            theta_ok && rho_ok
        });
        assert!(found, "Should find horizontal line near theta=π/2, rho=25. Found: {:?}", &peaks[..peaks.len().min(5)]);
    }

    #[test]
    fn test_hough_vertical_line() {
        let mut img = GrayImage::new(50, 50);
        for y in 0..50 {
            img.set(25, y, 255.0).unwrap();
        }
        let mut detector = HoughLineDetector::new(50, 50, 180, 100);
        detector.detect(&img);
        let peaks = detector.peaks(10, 5);
        assert!(!peaks.is_empty());
    }

    #[test]
    fn test_hough_no_lines() {
        let img = GrayImage::new(20, 20);
        let mut detector = HoughLineDetector::new(20, 20, 180, 100);
        detector.detect(&img);
        let peaks = detector.peaks(10, 5);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_affine_identity() {
        let mut img = GrayImage::new(5, 5);
        img.set(2, 2, 100.0).unwrap();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&img, &identity);
        assert!((result.get(2, 2).unwrap() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_affine_translation() {
        let mut img = GrayImage::new(10, 10);
        img.set(2, 2, 200.0).unwrap();
        let m = translation_matrix(2.0, 0.0);
        let result = affine_transform(&img, &m);
        // Pixel should move from (2,2) to (4,2)
        assert!(result.get(4, 2).unwrap() > 100.0);
    }

    #[test]
    fn test_affine_rotation_90() {
        let mut img = GrayImage::new(10, 10);
        img.set(5, 2, 255.0).unwrap();
        let m = rotation_matrix(std::f64::consts::FRAC_PI_2, 5.0, 5.0);
        let result = affine_transform(&img, &m);
        // After 90° rotation around (5,5): (5,2) → (8,5)
        let val = result.get(8, 5).unwrap();
        assert!(val > 100.0, "Expected high value at (8,5), got {}", val);
    }

    #[test]
    fn test_scale_matrix() {
        let m = scale_matrix(2.0, 2.0, 5.0, 5.0);
        assert_eq!(m[0][0], 2.0);
        assert_eq!(m[1][1], 2.0);
    }

    #[test]
    fn test_hough_accumulator_access() {
        let mut detector = HoughLineDetector::new(10, 10, 90, 50);
        assert_eq!(detector.get(0, 0), 0);
    }
}
