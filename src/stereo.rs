//! Stereo vision basics — disparity, triangulation.

use crate::image::GrayImage;
use nalgebra::{Matrix3, Vector3};

/// A stereo camera rig with baseline and focal length.
#[derive(Debug, Clone)]
pub struct StereoRig {
    pub baseline: f64,   // distance between cameras (meters)
    pub focal_length: f64, // in pixels
    pub cx: f64,         // principal point x
    pub cy: f64,         // principal point y
}

impl StereoRig {
    pub fn new(baseline: f64, focal_length: f64, cx: f64, cy: f64) -> Self {
        Self { baseline, focal_length, cx, cy }
    }

    /// Compute disparity map using block matching (SSD).
    pub fn compute_disparity(&self, left: &GrayImage, right: &GrayImage, max_disparity: usize, block_size: usize) -> GrayImage {
        let w = left.width();
        let h = left.height();
        let half = block_size / 2;
        let mut disp = GrayImage::new(w, h);

        for y in half..h.saturating_sub(half) {
            for x in half..w.saturating_sub(half) {
                let mut best_d = 0usize;
                let mut best_cost = f64::MAX;

                for d in 0..max_disparity.min(x) {
                    let mut cost = 0.0f64;
                    for dy in -(half as isize)..=(half as isize) {
                        for dx in -(half as isize)..=(half as isize) {
                            let lx = (x as isize + dx) as usize;
                            let ly = (y as isize + dy) as usize;
                            let rx = (x as isize + dx - d as isize) as usize;
                            let ry = ly;
                            let diff = left.data[ly * w + lx] - right.data[ry * w + rx];
                            cost += diff * diff;
                        }
                    }
                    if cost < best_cost {
                        best_cost = cost;
                        best_d = d;
                    }
                }
                disp.data_mut()[y * w + x] = best_d as f64;
            }
        }
        disp
    }

    /// Triangulate a 3D point from disparity.
    pub fn triangulate(&self, x_left: f64, y_left: f64, disparity: f64) -> Option<Vector3<f64>> {
        if disparity <= 0.0 { return None; }
        let z = self.baseline * self.focal_length / disparity;
        let x = (x_left - self.cx) * z / self.focal_length;
        let y = (y_left - self.cy) * z / self.focal_length;
        Some(Vector3::new(x, y, z))
    }

    /// Convert disparity map to depth map.
    pub fn disparity_to_depth(&self, disparity: &GrayImage) -> GrayImage {
        disparity.map(|d| {
            if d <= 0.0 { 0.0 } else { self.baseline * self.focal_length / d }
        })
    }

    /// Get the camera intrinsic matrix.
    pub fn intrinsic_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.focal_length, 0.0, self.cx,
            0.0, self.focal_length, self.cy,
            0.0, 0.0, 1.0,
        )
    }
}

/// Reproject disparity map to 3D point cloud.
pub fn reproject_to_3d(rig: &StereoRig, disparity: &GrayImage) -> Vec<Option<Vector3<f64>>> {
    let mut points = Vec::with_capacity(disparity.width() * disparity.height());
    for y in 0..disparity.height() {
        for x in 0..disparity.width() {
            let d = disparity.data[y * disparity.width() + x];
            points.push(rig.triangulate(x as f64, y as f64, d));
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stereo_pair() -> (GrayImage, GrayImage) {
        let mut left = GrayImage::new(30, 20);
        let mut right = GrayImage::new(30, 20);

        // A bright square at (10,5)-(14,9) in left, shifted by 3 in right
        for y in 5..10 {
            for x in 10..15 {
                left.set(x, y, 200.0).unwrap();
                right.set(x - 3, y, 200.0).unwrap();
            }
        }
        (left, right)
    }

    #[test]
    fn test_stereo_disparity() {
        let (left, right) = make_stereo_pair();
        let rig = StereoRig::new(0.1, 500.0, 15.0, 10.0);
        let disp = rig.compute_disparity(&left, &right, 10, 3);
        // In the square region, disparity should be ~3
        let d = disp.get(12, 7).unwrap();
        assert!((d - 3.0).abs() < 1.5, "Expected disparity ~3, got {}", d);
    }

    #[test]
    fn test_triangulation() {
        let rig = StereoRig::new(0.12, 600.0, 320.0, 240.0);
        let point = rig.triangulate(320.0, 240.0, 60.0).unwrap();
        let z = 0.12 * 600.0 / 60.0;
        assert!((point.z - z).abs() < 0.01);
        // At principal point, x and y should be 0
        assert!(point.x.abs() < 0.01);
        assert!(point.y.abs() < 0.01);
    }

    #[test]
    fn test_triangulation_zero_disparity() {
        let rig = StereoRig::new(0.1, 500.0, 250.0, 250.0);
        assert!(rig.triangulate(100.0, 100.0, 0.0).is_none());
    }

    #[test]
    fn test_disparity_to_depth() {
        let rig = StereoRig::new(0.1, 500.0, 250.0, 250.0);
        let mut disp = GrayImage::new(3, 1);
        disp.set(0, 0, 0.0).unwrap();
        disp.set(1, 0, 50.0).unwrap();
        disp.set(2, 0, 100.0).unwrap();
        let depth = rig.disparity_to_depth(&disp);
        assert_eq!(depth.get(0, 0).unwrap(), 0.0); // invalid disparity
        assert!((depth.get(1, 0).unwrap() - 1.0).abs() < 0.01); // 0.1*500/50
        assert!((depth.get(2, 0).unwrap() - 0.5).abs() < 0.01); // 0.1*500/100
    }

    #[test]
    fn test_intrinsic_matrix() {
        let rig = StereoRig::new(0.1, 500.0, 320.0, 240.0);
        let k = rig.intrinsic_matrix();
        assert_eq!(k[(0, 0)], 500.0);
        assert_eq!(k[(0, 2)], 320.0);
        assert_eq!(k[(1, 2)], 240.0);
        assert_eq!(k[(2, 2)], 1.0);
    }

    #[test]
    fn test_reproject_to_3d() {
        let rig = StereoRig::new(0.1, 500.0, 250.0, 250.0);
        let mut disp = GrayImage::new(2, 1);
        disp.set(0, 0, 0.0).unwrap();
        disp.set(1, 0, 50.0).unwrap();
        let points = reproject_to_3d(&rig, &disp);
        assert_eq!(points.len(), 2);
        assert!(points[0].is_none());
        assert!(points[1].is_some());
    }
}
