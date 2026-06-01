# lau-computer-vision

> A teaching-quality Rust library covering the core of a computer-vision course — grayscale image processing, convolution filters, morphological operations, feature detection, segmentation, geometric transforms, stereo vision, and optical flow — all fully tested.

**67 tests · zero unsafe · serde-serializable · `cargo add lau-computer-vision`**

---

## What This Does

`lau-computer-vision` implements the fundamental algorithms taught in a first course on computer vision, designed for agent visual perception:

| Module | What it covers |
|---|---|
| `image` | Grayscale image type with pixel ops, histograms, Otsu thresholding, convolution |
| `filter` | Gaussian/box blur, sharpen, Sobel/Prewitt gradients, Canny edge detection, Laplacian |
| `morphology` | Erosion, dilation, opening, closing, gradient, top-hat, black-hat with configurable structuring elements |
| `feature` | Harris corner detection, Laplacian-of-Gaussian blob detection |
| `segmentation` | Thresholding (binary, range, adaptive), connected-component labelling, watershed |
| `transform` | Hough line detection, affine image transforms with bilinear interpolation |
| `stereo` | Stereo disparity via block matching (SSD), triangulation, depth maps, point clouds |
| `optical_flow` | Lucas-Kanade dense optical flow |

---

## The Key Idea

Computer vision turns **pixel arrays** into **understanding**. This crate walks through the classic low- and mid-level pipeline:

1. **Image representation** — a grayscale `GrayImage` backed by `Vec<f64>`, with bounds-checked pixel access, boundary modes (zero-pad, clamp-to-edge), and generic `map`/`zip_with` operations.
2. **Filtering** — convolution as the universal operation: blur to suppress noise, sharpen to enhance edges, Sobel/Prewitt to compute gradients, Canny for clean edge maps.
3. **Morphology** — shape-based operations on binary/gray images: grow (dilate), shrink (erode), fill holes (close), remove specks (open), extract outlines (gradient).
4. **Feature detection** — find distinctive points in the image: Harris corners via the structure tensor, blobs via multi-scale Laplacian of Gaussian.
5. **Segmentation** — partition the image into meaningful regions: thresholding (fixed, adaptive, Otsu-automatic), connected components with union-find, watershed from seed markers.
6. **Geometric vision** — detect geometric primitives: Hough transform for lines; affine image warping with bilinear interpolation.
7. **3D and motion** — stereo depth from disparity maps, Lucas-Kanade optical flow for pixel-level motion estimation.

---

## Install

```toml
[dependencies]
lau-computer-vision = "0.1"
```

```bash
cargo add lau-computer-vision
```

### Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` 0.33 | Vectors and matrices (stereo rig, 3D triangulation) |
| `serde` (derive) | Serialisable image data |

---

## Quick Start

```rust
use lau_computer_vision::{GrayImage, ImageError};
use lau_computer_vision::filter;
use lau_computer_vision::feature;
use lau_computer_vision::segmentation;

// Create an image
let mut img = GrayImage::new(100, 100);
for y in 0..100 {
    for x in 0..50 {
        img.set(x, y, 200.0).unwrap(); // left half bright
    }
}

// Gaussian blur
let blurred = filter::gaussian_blur(&img, 1.5);

// Sobel edge detection
let (magnitude, direction, gx, gy) = filter::sobel(&blurred);

// Canny edges
let edges = filter::canny(&img, 20.0, 60.0);

// Harris corners
let corners = feature::harris_corners(&img, 0.04, 1.0, 3);

// Otsu automatic threshold
let thresh = img.otsu_threshold();
let binary = segmentation::threshold(&img, thresh);

// Connected components
let components = segmentation::connected_components(&binary);
for comp in &components {
    println!("Component {}: area={}, centroid=({:.1}, {:.1})",
        comp.label, comp.area, comp.centroid.0, comp.centroid.1);
}
```

---

## API Reference

### `image` — Grayscale image type

```rust
// Construction
let img = GrayImage::new(width, height);
let img = GrayImage::from_vec(w, h, vec![...f64])?;
let img = GrayImage::from_u8(w, h, &bytes)?;

// Pixel access
img.get(x, y) -> Result<f64, ImageError>
img.get_padded(x: isize, y: isize) -> f64      // zero-padded boundary
img.get_clamped(x: isize, y: isize) -> f64     // clamp-to-edge boundary
img.set(x, y, val) -> Result<(), ImageError>

// Statistics
img.histogram() -> [usize; 256]
img.histogram_normalized() -> [f64; 256]
img.mean() -> f64
img.std_dev() -> f64
img.otsu_threshold() -> f64

// Transforms
img.map(|v| v * 2.0) -> GrayImage
img.zip_with(&other, |a, b| a + b) -> Result<GrayImage, ImageError>
img.convolve(&kernel) -> Result<GrayImage, ImageError>
img.convolve_separable(&h_kernel, &v_kernel) -> Result<GrayImage, ImageError>
```

### `filter` — Convolution filters

```rust
// Blur
filter::gaussian_blur(&img, sigma) -> GrayImage
filter::box_blur(&img, size) -> GrayImage

// Edge detection
filter::sobel(&img) -> (magnitude, direction, gx, gy)  // all GrayImages
filter::canny(&img, low_threshold, high_threshold) -> GrayImage
filter::laplacian(&img) -> GrayImage
filter::sharpen(&img) -> GrayImage

// Raw kernels
filter::gaussian_kernel(sigma, size) -> Vec<Vec<f64>>
filter::sobel_gx() / sobel_gy() -> Vec<Vec<f64>>
filter::prewitt_gx() / prewitt_gy() -> Vec<Vec<f64>>
filter::laplacian_kernel() -> Vec<Vec<f64>>
filter::sharpen_kernel() -> Vec<Vec<f64>>
```

### `morphology` — Shape operations

```rust
// Structuring elements
let se = StructuringElement::square(3);
let se = StructuringElement::cross(5);

// Operations
morphology::erode(&img, &se) -> GrayImage
morphology::dilate(&img, &se) -> GrayImage
morphology::open(&img, &se) -> GrayImage      // erode then dilate
morphology::close(&img, &se) -> GrayImage     // dilate then erode
morphology::gradient(&img, &se) -> GrayImage  // dilate - erode
morphology::top_hat(&img, &se) -> GrayImage   // image - opening
morphology::black_hat(&img, &se) -> GrayImage // closing - image
```

### `feature` — Corner and blob detection

```rust
// Harris corners
let response = feature::harris_response(&img, k);
let corners = feature::harris_corners(&img, k, threshold, nms_radius) -> Vec<FeaturePoint>;

// Blob detection (multi-scale LoG)
let blobs = feature::detect_blobs(&img, min_sigma, max_sigma, num_scales, threshold) -> Vec<FeaturePoint>;

// FeaturePoint fields: x, y, response (strength), scale
```

### `segmentation` — Region-based image partitioning

```rust
// Thresholding
segmentation::threshold(&img, t) -> GrayImage
segmentation::threshold_range(&img, low, high) -> GrayImage
segmentation::threshold_inverse(&img, t) -> GrayImage
segmentation::adaptive_threshold(&img, block_size, c) -> GrayImage

// Connected components (4-connected, union-find)
let components = segmentation::connected_components(&binary_img) -> Vec<Component>;
// Component fields: label, pixels, area, centroid (f64,f64), bounding_box

// Watershed
segmentation::watershed(&markers, &gradient) -> GrayImage
```

### `transform` — Geometric image transforms

```rust
// Hough line detection
let mut detector = HoughLineDetector::new(width, height, theta_bins, rho_bins);
detector.detect(&edge_image);
let lines = detector.peaks(threshold, nms_radius) -> Vec<HoughLine>;
// HoughLine fields: rho, theta, votes

// Affine transforms (2×3 matrix, inverse mapping, bilinear interpolation)
let result = transform::affine_transform(&img, &matrix);
transform::rotation_matrix(angle, cx, cy) -> [[f64; 3]; 2]
transform::translation_matrix(tx, ty) -> [[f64; 3]; 2]
transform::scale_matrix(sx, sy, cx, cy) -> [[f64; 3]; 2]
```

### `stereo` — Stereo depth perception

```rust
let rig = StereoRig::new(baseline, focal_length, cx, cy);

// Disparity via block matching (SSD)
let disparity = rig.compute_disparity(&left, &right, max_disparity, block_size);

// Triangulate 3D point
let point = rig.triangulate(x_left, y_left, disparity) -> Option<Vector3<f64>>;

// Depth map
let depth = rig.disparity_to_depth(&disparity);

// Point cloud
let points = reproject_to_3d(&rig, &disparity) -> Vec<Option<Vector3<f64>>>;

// Camera intrinsics
let k = rig.intrinsic_matrix() -> Matrix3<f64>;
```

### `optical_flow` — Motion estimation

```rust
// Lucas-Kanade dense flow (magnitude image)
let flow_mag = optical_flow::lucas_kanade(&prev, &curr, window_size) -> GrayImage;

// With separate u/v components
let (flow_u, flow_v) = optical_flow::lucas_kanade_uv(&prev, &curr, window_size);

// Flow vectors
let flow = optical_flow::dense_flow(&prev, &curr, window_size) -> Vec<Vec<Option<FlowVector>>>;
// FlowVector { u: f64, v: f64 }
```

---

## How It Works

### Convolution

Every linear filter is implemented as a 2D convolution: slide a kernel across the image, multiply element-wise, and sum. The `GrayImage::convolve` method handles boundary conditions via zero-padding. Separable kernels (Gaussian) can use `convolve_separable` for O(n·k) instead of O(n·k²).

### Canny Edge Detection

1. **Gaussian blur** — suppress noise (default σ = 1.4).
2. **Sobel gradients** — compute magnitude and direction at each pixel.
3. **Non-maximum suppression** — thin edges to 1-pixel wide by keeping only the maximum along the gradient direction.
4. **Double threshold + hysteresis** — strong edges (> high) are kept; weak edges (> low) are kept only if connected to a strong edge.

### Harris Corner Detection

Compute the structure tensor at each pixel:

$$M = \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix}$$

where $I_x, I_y$ are image gradients (Sobel), smoothed with a Gaussian window. The Harris response is:

$$R = \det(M) - k \cdot \text{trace}(M)^2$$

Corners have large R (both eigenvalues large), edges have negative R (one eigenvalue dominant), flat regions have near-zero R. Non-maximum suppression selects isolated peaks.

### Connected Components

Uses a classic two-pass algorithm with union-find:
1. **First pass** — scan left-to-right, top-to-bottom; assign provisional labels; merge neighbours via union-find.
2. **Second pass** — resolve all labels to their root, group pixels into components, compute area, centroid, and bounding box.

### Hough Transform

For each edge pixel $(x, y)$, vote in $(\rho, \theta)$ parameter space for all lines that could pass through it: $\rho = x\cos\theta + y\sin\theta$. Peaks in the accumulator correspond to detected lines. Non-maximum suppression in parameter space selects clean peaks.

### Lucas-Kanade Optical Flow

Assume the brightness constancy constraint: $I_x u + I_y v + I_t = 0$ (one equation per pixel). In a small window around each pixel, accumulate:

$$\sum I_x^2, \quad \sum I_y^2, \quad \sum I_x I_y, \quad \sum I_x I_t, \quad \sum I_y I_t$$

Solve the 2×2 system via Cramer's rule. Where the determinant is near zero (aperture problem), no flow is reported.

### Stereo Disparity

Block matching with sum-of-squared-differences (SSD): for each pixel in the left image, search along the epipolar line (same row) in the right image for the best-matching block. The displacement is the disparity. Triangulation gives depth: $Z = \frac{b \cdot f}{d}$ where $b$ is baseline, $f$ is focal length, $d$ is disparity.

---

## The Math

### Otsu's Threshold

Maximise the between-class variance $\sigma_B^2(t) = w_0(t) \cdot w_1(t) \cdot (\mu_0(t) - \mu_1(t))^2$ over all thresholds $t$. This is equivalent to minimising within-class variance and produces an optimal threshold for bimodal histograms.

### Laplacian of Gaussian (LoG) for Blob Detection

The LoG is $\nabla^2 G(x, y, \sigma)$ — a Gaussian-smoothed Laplacian. Blobs produce strong responses at a scale proportional to their size. Scale-normalised response $\sigma^2 |LoG|$ is compared across scales to find the characteristic scale of each blob.

### Affine Transform with Bilinear Interpolation

For an affine map $\mathbf{x}' = A\mathbf{x} + \mathbf{t}$, we use **inverse mapping**: for each output pixel, compute the source coordinate via $A^{-1}$, then interpolate the source image with bilinear interpolation:

$$f(x, y) \approx f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy$$

This avoids holes in the output and produces smooth results.

---

## Testing

**67 unit tests** covering:

- **Image:** creation, pixel access, boundaries, histograms, mean/std, Otsu, convolution identity, map, zip_with, u8 conversion
- **Filters:** kernel normalisation (sums to 1), brightness preservation, edge response, sharpening, Canny edge detection, Laplacian zero-sum kernel
- **Morphology:** erosion shrinks, dilation grows, opening removes noise, closing fills gaps, gradient outlines, structuring elements
- **Features:** Harris corner detection (4+ corners on checkerboard), uniform image zero response, edge negative response, blob detection
- **Segmentation:** threshold types, adaptive threshold, connected components (count, centroid, bounding box, empty), watershed
- **Transforms:** Hough line detection (horizontal, vertical, empty), affine identity/translation/rotation, matrix helpers
- **Stereo:** disparity matching, triangulation geometry, depth conversion, intrinsic matrix, point cloud reprojection
- **Optical flow:** static image zero flow, motion detection, magnitude, uniform image, dense flow structure

```bash
cargo test
```

---

## License

MIT
