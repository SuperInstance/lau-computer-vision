//! Image basics — grayscale images, pixel operations, histograms.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error type for image operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageError {
    OutOfBounds { x: usize, y: usize, width: usize, height: usize },
    EmptyImage,
    InvalidDimensions,
    DimensionMismatch,
    InvalidKernel,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "({},{}) out of bounds for {}x{} image", x, y, width, height)
            }
            Self::EmptyImage => write!(f, "empty image"),
            Self::InvalidDimensions => write!(f, "invalid dimensions"),
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::InvalidKernel => write!(f, "invalid kernel"),
        }
    }
}

impl std::error::Error for ImageError {}

/// A grayscale image with 8-bit pixels (0–255).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrayImage {
    width: usize,
    height: usize,
    pub data: Vec<f64>,
}

impl GrayImage {
    /// Create a new blank (all zeros) grayscale image.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// Create an image from raw f64 data (row-major).
    pub fn from_vec(width: usize, height: usize, data: Vec<f64>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions);
        }
        if data.len() != width * height {
            return Err(ImageError::DimensionMismatch);
        }
        Ok(Self { width, height, data })
    }

    /// Create from u8 pixel data, converting to f64.
    pub fn from_u8(width: usize, height: usize, data: &[u8]) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions);
        }
        if data.len() != width * height {
            return Err(ImageError::DimensionMismatch);
        }
        Ok(Self {
            width,
            height,
            data: data.iter().map(|&v| v as f64).collect(),
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    /// Get pixel value at (x, y).
    pub fn get(&self, x: usize, y: usize) -> Result<f64, ImageError> {
        if x >= self.width || y >= self.height {
            Err(ImageError::OutOfBounds {
                x, y, width: self.width, height: self.height,
            })
        } else {
            Ok(self.data[y * self.width + x])
        }
    }

    /// Set pixel value at (x, y).
    pub fn set(&mut self, x: usize, y: usize, val: f64) -> Result<(), ImageError> {
        if x >= self.width || y >= self.height {
            Err(ImageError::OutOfBounds {
                x, y, width: self.width, height: self.height,
            })
        } else {
            self.data[y * self.width + x] = val;
            Ok(())
        }
    }

    /// Get pixel with zero-padding boundary handling.
    pub fn get_padded(&self, x: isize, y: isize) -> f64 {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            0.0
        } else {
            self.data[y as usize * self.width + x as usize]
        }
    }

    /// Get pixel with clamp-to-edge boundary handling.
    pub fn get_clamped(&self, x: isize, y: isize) -> f64 {
        let cx = x.clamp(0, (self.width - 1) as isize) as usize;
        let cy = y.clamp(0, (self.height - 1) as isize) as usize;
        self.data[cy * self.width + cx]
    }

    /// Get the raw data slice.
    pub fn data(&self) -> &[f64] { &self.data }

    /// Get mutable raw data.
    pub fn data_mut(&mut self) -> &mut Vec<f64> { &mut self.data }

    /// Convert pixel to u8, clamping to [0, 255].
    pub fn to_u8(&self) -> Vec<u8> {
        self.data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect()
    }

    /// Compute the histogram (256 bins).
    pub fn histogram(&self) -> [usize; 256] {
        let mut hist = [0usize; 256];
        for &val in &self.data {
            let bin = (val.clamp(0.0, 255.0) as usize).min(255);
            hist[bin] += 1;
        }
        hist
    }

    /// Compute the normalized histogram (each bin is a proportion).
    pub fn histogram_normalized(&self) -> [f64; 256] {
        let hist = self.histogram();
        let total = self.data.len() as f64;
        let mut norm = [0.0f64; 256];
        for i in 0..256 {
            norm[i] = hist[i] as f64 / total;
        }
        norm
    }

    /// Compute Otsu's threshold.
    pub fn otsu_threshold(&self) -> f64 {
        let hist = self.histogram_normalized();
        let mut best_threshold = 0.0f64;
        let mut best_variance = 0.0f64;

        let mu_total: f64 = (0..256).map(|i| i as f64 * hist[i]).sum();

        let mut w0 = 0.0f64;
        let mut mu0 = 0.0f64;

        for t in 0..256 {
            w0 += hist[t];
            if w0 == 0.0 { continue; }
            let w1 = 1.0 - w0;
            if w1 == 0.0 { break; }

            mu0 += t as f64 * hist[t];
            let mu0_avg = mu0 / w0;
            let mu1_avg = (mu_total - mu0) / w1;

            let variance = w0 * w1 * (mu0_avg - mu1_avg).powi(2);
            if variance > best_variance {
                best_variance = variance;
                best_threshold = t as f64;
            }
        }
        best_threshold
    }

    /// Compute mean pixel value.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Compute standard deviation of pixel values.
    pub fn std_dev(&self) -> f64 {
        let m = self.mean();
        let variance = self.data.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / self.data.len() as f64;
        variance.sqrt()
    }

    /// Map every pixel through a function.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Element-wise arithmetic on two images.
    pub fn zip_with(&self, other: &GrayImage, f: impl Fn(f64, f64) -> f64) -> Result<GrayImage, ImageError> {
        if self.width != other.width || self.height != other.height {
            return Err(ImageError::DimensionMismatch);
        }
        Ok(GrayImage {
            width: self.width,
            height: self.height,
            data: self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect(),
        })
    }

    /// Apply a 2D convolution kernel.
    pub fn convolve(&self, kernel: &[Vec<f64>]) -> Result<GrayImage, ImageError> {
        let kh = kernel.len();
        if kh == 0 { return Err(ImageError::InvalidKernel); }
        let kw = kernel[0].len();
        if kw == 0 { return Err(ImageError::InvalidKernel); }
        // Verify rectangular
        for row in kernel {
            if row.len() != kw { return Err(ImageError::InvalidKernel); }
        }

        let ky = kh as isize / 2;
        let kx = kw as isize / 2;

        let mut out = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0f64;
                for (kyi, krow) in kernel.iter().enumerate() {
                    for (kxi, &kval) in krow.iter().enumerate() {
                        let ix = x as isize + kxi as isize - kx;
                        let iy = y as isize + kyi as isize - ky;
                        sum += self.get_padded(ix, iy) * kval;
                    }
                }
                out.data[y * self.width + x] = sum;
            }
        }
        Ok(out)
    }

    /// Apply a separable convolution (1D horizontal then 1D vertical).
    pub fn convolve_separable(&self, kernel_h: &[f64], kernel_v: &[f64]) -> Result<GrayImage, ImageError> {
        if kernel_h.is_empty() || kernel_v.is_empty() {
            return Err(ImageError::InvalidKernel);
        }
        // Horizontal pass
        let khr = kernel_h.len() as isize / 2;
        let mut temp = vec![0.0f64; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0;
                for (ki, &kv) in kernel_h.iter().enumerate() {
                    let ix = x as isize + ki as isize - khr;
                    sum += self.get_padded(ix, y as isize) * kv;
                }
                temp[y * self.width + x] = sum;
            }
        }
        // Vertical pass
        let kvr = kernel_v.len() as isize / 2;
        let mut out = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0;
                for (ki, &kv) in kernel_v.iter().enumerate() {
                    let iy = y as isize + ki as isize - kvr;
                    let iy_c = iy.clamp(0, (self.height - 1) as isize) as usize;
                    sum += temp[iy_c * self.width + x] * kv;
                }
                out.data[y * self.width + x] = sum;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_blank_image() {
        let img = GrayImage::new(10, 20);
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 20);
        assert_eq!(img.get(5, 5).unwrap(), 0.0);
    }

    #[test]
    fn test_set_and_get() {
        let mut img = GrayImage::new(5, 5);
        img.set(2, 3, 128.0).unwrap();
        assert_eq!(img.get(2, 3).unwrap(), 128.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let img = GrayImage::new(5, 5);
        assert!(img.get(5, 0).is_err());
        assert!(img.get(0, 5).is_err());
    }

    #[test]
    fn test_histogram() {
        let mut img = GrayImage::new(2, 2);
        img.set(0, 0, 0.0).unwrap();
        img.set(1, 0, 100.0).unwrap();
        img.set(0, 1, 100.0).unwrap();
        img.set(1, 1, 200.0).unwrap();
        let hist = img.histogram();
        assert_eq!(hist[0], 1);
        assert_eq!(hist[100], 2);
        assert_eq!(hist[200], 1);
    }

    #[test]
    fn test_mean_std() {
        let mut img = GrayImage::new(2, 2);
        img.set(0, 0, 0.0).unwrap();
        img.set(1, 0, 100.0).unwrap();
        img.set(0, 1, 100.0).unwrap();
        img.set(1, 1, 200.0).unwrap();
        assert_eq!(img.mean(), 100.0);
    }

    #[test]
    fn test_from_u8() {
        let data = vec![0u8, 128, 255, 64];
        let img = GrayImage::from_u8(2, 2, &data).unwrap();
        assert_eq!(img.get(1, 0).unwrap(), 128.0);
        assert_eq!(img.get(1, 1).unwrap(), 64.0);
    }

    #[test]
    fn test_padded_boundary() {
        let mut img = GrayImage::new(3, 3);
        img.set(1, 1, 42.0).unwrap();
        assert_eq!(img.get_padded(-1, -1), 0.0);
        assert_eq!(img.get_padded(3, 3), 0.0);
        assert_eq!(img.get_padded(1, 1), 42.0);
    }

    #[test]
    fn test_clamped_boundary() {
        let mut img = GrayImage::new(3, 3);
        img.set(0, 0, 10.0).unwrap();
        img.set(2, 2, 99.0).unwrap();
        assert_eq!(img.get_clamped(-1, -1), 10.0);
        assert_eq!(img.get_clamped(3, 3), 99.0);
    }

    #[test]
    fn test_map() {
        let mut img = GrayImage::new(2, 2);
        img.set(0, 0, 10.0).unwrap();
        img.set(1, 0, 20.0).unwrap();
        let doubled = img.map(|v| v * 2.0);
        assert_eq!(doubled.get(0, 0).unwrap(), 20.0);
        assert_eq!(doubled.get(1, 0).unwrap(), 40.0);
    }

    #[test]
    fn test_zip_with() {
        let mut a = GrayImage::new(2, 2);
        let mut b = GrayImage::new(2, 2);
        a.set(0, 0, 10.0).unwrap();
        b.set(0, 0, 5.0).unwrap();
        let c = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(c.get(0, 0).unwrap(), 15.0);
    }

    #[test]
    fn test_to_u8_clamping() {
        let mut img = GrayImage::new(2, 1);
        img.set(0, 0, -10.0).unwrap();
        img.set(1, 0, 300.0).unwrap();
        let bytes = img.to_u8();
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 255);
    }

    #[test]
    fn test_otsu_bimodal() {
        // Left half dark, right half bright
        let mut img = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let val = if x < 50 { 30.0 } else { 200.0 };
                img.set(x, y, val).unwrap();
            }
        }
        let t = img.otsu_threshold();
        assert!(t > 20.0 && t < 190.0, "Otsu threshold {} should be between the two modes", t);
    }

    #[test]
    fn test_convolution_identity() {
        let mut img = GrayImage::new(5, 5);
        img.set(2, 2, 100.0).unwrap();
        let kernel = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let result = img.convolve(&kernel).unwrap();
        assert_eq!(result.get(2, 2).unwrap(), 100.0);
    }
}
