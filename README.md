# lau-computer-vision

> Computer vision fundamentals — image processing, feature detection, and geometric vision for agent visual perception

## What This Does

Computer vision fundamentals — image processing, feature detection, and geometric vision for agent visual perception. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-computer-vision
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_computer_vision::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct FlowVector 
pub fn lucas_kanade(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> GrayImage 
pub fn lucas_kanade_uv(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> (GrayImage, GrayImage) 
pub fn dense_flow(prev: &GrayImage, curr: &GrayImage, window_size: usize) -> Vec<Vec<Option<FlowVector>>> 
pub struct HoughLine 
pub struct HoughLineDetector 
    pub fn new(width: usize, height: usize, theta_bins: usize, rho_bins: usize) -> Self 
    pub fn detect(&mut self, edge_image: &GrayImage) 
    pub fn get(&self, theta_bin: usize, rho_bin: usize) -> usize 
    pub fn peaks(&self, threshold: usize, nms_radius: usize) -> Vec<HoughLine> 
pub fn affine_transform(img: &GrayImage, matrix: &[[f64; 3]; 2]) -> GrayImage 
pub fn rotation_matrix(angle_rad: f64, cx: f64, cy: f64) -> [[f64; 3]; 2] 
pub fn translation_matrix(tx: f64, ty: f64) -> [[f64; 3]; 2] 
pub fn scale_matrix(sx: f64, sy: f64, cx: f64, cy: f64) -> [[f64; 3]; 2] 
pub fn threshold(img: &GrayImage, thresh: f64) -> GrayImage 
pub fn threshold_range(img: &GrayImage, low: f64, high: f64) -> GrayImage 
pub fn threshold_inverse(img: &GrayImage, thresh: f64) -> GrayImage 
pub fn adaptive_threshold(img: &GrayImage, block_size: usize, c: f64) -> GrayImage 
pub struct Component 
pub fn connected_components(binary: &GrayImage) -> Vec<Component> 
pub fn watershed(markers: &GrayImage, gradient: &GrayImage) -> GrayImage 
pub struct FeaturePoint 
pub fn harris_response(img: &GrayImage, k: f64) -> GrayImage 
pub fn harris_corners(img: &GrayImage, k: f64, threshold: f64, nms_radius: usize) -> Vec<FeaturePoint> 
pub fn detect_blobs(img: &GrayImage, min_sigma: f64, max_sigma: f64, num_scales: usize, threshold: f64) -> Vec<FeaturePoint> 
pub enum ImageError 
pub struct GrayImage 
    pub fn new(width: usize, height: usize) -> Self 
    pub fn from_vec(width: usize, height: usize, data: Vec<f64>) -> Result<Self, ImageError> 
    pub fn from_u8(width: usize, height: usize, data: &[u8]) -> Result<Self, ImageError> 
    pub fn width(&self) -> usize  self.width }
    pub fn height(&self) -> usize  self.height }
    pub fn get(&self, x: usize, y: usize) -> Result<f64, ImageError> 
    pub fn set(&mut self, x: usize, y: usize, val: f64) -> Result<(), ImageError> 
    pub fn get_padded(&self, x: isize, y: isize) -> f64 
    pub fn get_clamped(&self, x: isize, y: isize) -> f64 
    pub fn data(&self) -> &[f64]  &self.data }
    pub fn data_mut(&mut self) -> &mut Vec<f64>  &mut self.data }
    pub fn to_u8(&self) -> Vec<u8> 
    pub fn histogram(&self) -> [usize; 256] 
    pub fn histogram_normalized(&self) -> [f64; 256] 
    pub fn otsu_threshold(&self) -> f64 
    pub fn mean(&self) -> f64 
    pub fn std_dev(&self) -> f64 
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self 
    pub fn zip_with(&self, other: &GrayImage, f: impl Fn(f64, f64) -> f64) -> Result<GrayImage, ImageError> 
    pub fn convolve(&self, kernel: &[Vec<f64>]) -> Result<GrayImage, ImageError> 
    pub fn convolve_separable(&self, kernel_h: &[f64], kernel_v: &[f64]) -> Result<GrayImage, ImageError> 
pub struct StructuringElement 
    pub fn square(size: usize) -> Self 
    pub fn cross(size: usize) -> Self 
    pub fn from_vec(data: Vec<Vec<bool>>) -> Self 
    pub fn width(&self) -> usize  self.width }
    pub fn height(&self) -> usize  self.height }
    pub fn get(&self, x: usize, y: usize) -> bool 
pub fn erode(img: &GrayImage, se: &StructuringElement) -> GrayImage 
pub fn dilate(img: &GrayImage, se: &StructuringElement) -> GrayImage 
pub fn open(img: &GrayImage, se: &StructuringElement) -> GrayImage 
pub fn close(img: &GrayImage, se: &StructuringElement) -> GrayImage 
pub fn gradient(img: &GrayImage, se: &StructuringElement) -> GrayImage 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**67 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
