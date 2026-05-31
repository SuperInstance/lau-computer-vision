//! Segmentation — thresholding, connected components.

use crate::image::GrayImage;

/// Apply binary thresholding.
pub fn threshold(img: &GrayImage, thresh: f64) -> GrayImage {
    img.map(|v| if v > thresh { 255.0 } else { 0.0 })
}

/// Apply binary thresholding with two values (band-pass).
pub fn threshold_range(img: &GrayImage, low: f64, high: f64) -> GrayImage {
    img.map(|v| if v >= low && v <= high { 255.0 } else { 0.0 })
}

/// Inverse threshold (below threshold → white).
pub fn threshold_inverse(img: &GrayImage, thresh: f64) -> GrayImage {
    img.map(|v| if v <= thresh { 255.0 } else { 0.0 })
}

/// Adaptive threshold using local mean.
pub fn adaptive_threshold(img: &GrayImage, block_size: usize, c: f64) -> GrayImage {
    let h = block_size / 2;
    let mut out = GrayImage::new(img.width(), img.height());

    for y in 0..img.height() {
        for x in 0..img.width() {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in -(h as isize)..=(h as isize) {
                for dx in -(h as isize)..=(h as isize) {
                    let nx = (x as isize + dx).clamp(0, (img.width() - 1) as isize) as usize;
                    let ny = (y as isize + dy).clamp(0, (img.height() - 1) as isize) as usize;
                    sum += img.data[ny * img.width() + nx];
                    count += 1;
                }
            }
            let local_mean = sum / count as f64;
            let val = img.data[y * img.width() + x];
            out.data_mut()[y * img.width() + x] = if val > local_mean - c { 255.0 } else { 0.0 };
        }
    }
    out
}

/// A labeled connected component.
#[derive(Debug, Clone)]
pub struct Component {
    pub label: usize,
    pub pixels: Vec<(usize, usize)>,
    pub area: usize,
    pub centroid: (f64, f64),
    pub bounding_box: (usize, usize, usize, usize), // min_x, min_y, max_x, max_y
}

/// Connected component labeling (4-connected) on a binary image.
pub fn connected_components(binary: &GrayImage) -> Vec<Component> {
    let w = binary.width();
    let h = binary.height();
    let mut labels = vec![0usize; w * h];
    let mut components: Vec<Vec<(usize, usize)>> = vec![vec![]]; // index 0 unused
    let mut next_label = 1usize;

    // Union-Find
    let mut parent = vec![0usize; w * h + 1];

    fn find(parent: &[usize], i: usize) -> usize {
        let mut x = i;
        while parent[x] != x {
            x = parent[x];
        }
        x
    }

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if binary.data[idx] == 0.0 { continue; }

            let mut neighbors = Vec::new();
            if x > 0 && labels[y * w + x - 1] > 0 {
                neighbors.push(labels[y * w + x - 1]);
            }
            if y > 0 && labels[(y - 1) * w + x] > 0 {
                neighbors.push(labels[(y - 1) * w + x]);
            }

            if neighbors.is_empty() {
                labels[idx] = next_label;
                parent.push(next_label);
                parent[next_label] = next_label;
                components.push(vec![(x, y)]);
                next_label += 1;
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[idx] = min_label;
                for &n in &neighbors {
                    let rn = find(&parent, n);
                    let rm = find(&parent, min_label);
                    if rn != rm {
                        parent[rn.max(rm)] = rn.min(rm);
                    }
                }
            }
        }
    }

    // Second pass — resolve labels
    let mut final_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next_final = 1usize;
    let mut pixel_groups: std::collections::HashMap<usize, Vec<(usize, usize)>> = std::collections::HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if labels[idx] == 0 { continue; }
            let root = find(&parent, labels[idx]);
            let final_label = *final_map.entry(root).or_insert_with(|| {
                let l = next_final;
                next_final += 1;
                l
            });
            pixel_groups.entry(final_label).or_default().push((x, y));
        }
    }

    let mut result: Vec<Component> = pixel_groups
        .into_iter()
        .map(|(label, pixels)| {
            let area = pixels.len();
            let sum_x: f64 = pixels.iter().map(|&(x, _)| x as f64).sum();
            let sum_y: f64 = pixels.iter().map(|&(_, y)| y as f64).sum();
            let centroid = (sum_x / area as f64, sum_y / area as f64);

            let min_x = pixels.iter().map(|&(x, _)| x).min().unwrap_or(0);
            let max_x = pixels.iter().map(|&(x, _)| x).max().unwrap_or(0);
            let min_y = pixels.iter().map(|&(_, y)| y).min().unwrap_or(0);
            let max_y = pixels.iter().map(|&(_, y)| y).max().unwrap_or(0);

            Component { label, pixels, area, centroid, bounding_box: (min_x, min_y, max_x, max_y) }
        })
        .collect();

    result.sort_by_key(|c| c.label);
    result
}

/// Watershed-like segmentation using iterative flooding (simplified).
pub fn watershed(markers: &GrayImage, gradient: &GrayImage) -> GrayImage {
    let w = markers.width();
    let h = markers.height();
    let mut result = markers.clone();

    // Simple iterative region growing from markers
    for _ in 0..((w + h).max(50)) {
        let mut changed = false;
        let prev = result.clone();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if prev.data[y * w + x] != 0.0 { continue; }

                // Check 4-neighbors for a label
                let mut best_label = 0.0f64;
                let mut best_grad = f64::MAX;
                for (dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    let neighbor_label = prev.data[ny * w + nx];
                    if neighbor_label > 0.0 {
                        let g = gradient.data[y * w + x];
                        if g < best_grad {
                            best_grad = g;
                            best_label = neighbor_label;
                        }
                    }
                }
                if best_label > 0.0 {
                    result.data_mut()[y * w + x] = best_label;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_binary() {
        let mut img = GrayImage::new(4, 1);
        img.set(0, 0, 10.0).unwrap();
        img.set(1, 0, 50.0).unwrap();
        img.set(2, 0, 100.0).unwrap();
        img.set(3, 0, 200.0).unwrap();
        let binary = threshold(&img, 75.0);
        assert_eq!(binary.get(0, 0).unwrap(), 0.0);
        assert_eq!(binary.get(1, 0).unwrap(), 0.0);
        assert_eq!(binary.get(2, 0).unwrap(), 255.0);
        assert_eq!(binary.get(3, 0).unwrap(), 255.0);
    }

    #[test]
    fn test_threshold_range() {
        let mut img = GrayImage::new(4, 1);
        img.set(0, 0, 10.0).unwrap();
        img.set(1, 0, 50.0).unwrap();
        img.set(2, 0, 100.0).unwrap();
        img.set(3, 0, 200.0).unwrap();
        let result = threshold_range(&img, 40.0, 120.0);
        assert_eq!(result.get(0, 0).unwrap(), 0.0);
        assert_eq!(result.get(1, 0).unwrap(), 255.0);
        assert_eq!(result.get(2, 0).unwrap(), 255.0);
        assert_eq!(result.get(3, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_threshold_inverse() {
        let mut img = GrayImage::new(2, 1);
        img.set(0, 0, 50.0).unwrap();
        img.set(1, 0, 150.0).unwrap();
        let result = threshold_inverse(&img, 100.0);
        assert_eq!(result.get(0, 0).unwrap(), 255.0);
        assert_eq!(result.get(1, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_connected_components_single_blob() {
        let mut img = GrayImage::new(10, 10);
        for y in 2..5 {
            for x in 2..5 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let components = connected_components(&img);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].area, 9);
    }

    #[test]
    fn test_connected_components_two_blobs() {
        let mut img = GrayImage::new(20, 10);
        // Left blob
        for y in 2..5 {
            for x in 1..4 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        // Right blob (separated)
        for y in 2..5 {
            for x in 10..13 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let components = connected_components(&img);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_connected_components_centroid() {
        let mut img = GrayImage::new(10, 10);
        for y in 3..6 {
            for x in 3..6 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let components = connected_components(&img);
        assert_eq!(components.len(), 1);
        let (cx, cy) = components[0].centroid;
        assert!((cx - 4.0).abs() < 0.1);
        assert!((cy - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_connected_components_bounding_box() {
        let mut img = GrayImage::new(10, 10);
        for y in 2..6 {
            for x in 3..7 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        let components = connected_components(&img);
        assert_eq!(components[0].bounding_box, (3, 2, 6, 5));
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut img = GrayImage::new(20, 20);
        // Left half dark, right half bright
        for y in 0..20 {
            for x in 0..10 {
                img.set(x, y, 50.0).unwrap();
            }
            for x in 10..20 {
                img.set(x, y, 200.0).unwrap();
            }
        }
        let result = adaptive_threshold(&img, 5, 10.0);
        // Right half should be mostly white
        let right_mean: f64 = (10..20).map(|x| result.get(x, 10).unwrap()).sum::<f64>() / 10.0;
        assert!(right_mean > 100.0);
    }

    #[test]
    fn test_watershed_simple() {
        // Create gradient image with a valley
        let mut gradient = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let dx = (x as f64 - 3.0).abs();
                let dy = (y as f64 - 3.0).abs();
                gradient.set(x, y, (dx + dy).min(5.0)).unwrap();
            }
        }

        let mut markers = GrayImage::new(10, 10);
        markers.set(3, 3, 1.0).unwrap(); // seed

        let result = watershed(&markers, &gradient);
        // The center should have the label
        assert!(result.get(3, 3).unwrap() > 0.0);
    }

    #[test]
    fn test_empty_connected_components() {
        let img = GrayImage::new(10, 10);
        let components = connected_components(&img);
        assert!(components.is_empty());
    }
}
