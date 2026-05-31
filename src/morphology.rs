//! Morphological operations — erosion, dilation, opening, closing.

use crate::image::GrayImage;

/// A structuring element (binary kernel).
#[derive(Debug, Clone)]
pub struct StructuringElement {
    data: Vec<Vec<bool>>,
    width: usize,
    height: usize,
}

impl StructuringElement {
    /// Create a square structuring element.
    pub fn square(size: usize) -> Self {
        Self {
            data: vec![vec![true; size]; size],
            width: size,
            height: size,
        }
    }

    /// Create a cross (plus-shaped) structuring element.
    pub fn cross(size: usize) -> Self {
        let mut data = vec![vec![false; size]; size];
        let center = size / 2;
        for i in 0..size {
            data[center][i] = true;
            data[i][center] = true;
        }
        Self { data, width: size, height: size }
    }

    /// Create a custom structuring element from bools.
    pub fn from_vec(data: Vec<Vec<bool>>) -> Self {
        let h = data.len();
        let w = if h > 0 { data[0].len() } else { 0 };
        Self { data, width: w, height: h }
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    /// Get value at kernel position.
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.data.get(y).and_then(|row| row.get(x)).copied().unwrap_or(false)
    }
}

/// Erosion — for each pixel, take the minimum in the neighborhood defined by the structuring element.
pub fn erode(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let kx = se.width() as isize / 2;
    let ky = se.height() as isize / 2;
    let mut out = GrayImage::new(img.width(), img.height());

    for y in 0..img.height() {
        for x in 0..img.width() {
            let mut min_val = f64::MAX;
            for sy in 0..se.height() {
                for sx in 0..se.width() {
                    if se.get(sx, sy) {
                        let ix = x as isize + sx as isize - kx;
                        let iy = y as isize + sy as isize - ky;
                        let v = img.get_padded(ix, iy);
                        min_val = min_val.min(v);
                    }
                }
            }
            out.data_mut()[y * img.width() + x] = min_val;
        }
    }
    out
}

/// Dilation — for each pixel, take the maximum in the neighborhood.
pub fn dilate(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let kx = se.width() as isize / 2;
    let ky = se.height() as isize / 2;
    let mut out = GrayImage::new(img.width(), img.height());

    for y in 0..img.height() {
        for x in 0..img.width() {
            let mut max_val = f64::MIN;
            for sy in 0..se.height() {
                for sx in 0..se.width() {
                    if se.get(sx, sy) {
                        let ix = x as isize + sx as isize - kx;
                        let iy = y as isize + sy as isize - ky;
                        let v = img.get_padded(ix, iy);
                        max_val = max_val.max(v);
                    }
                }
            }
            out.data_mut()[y * img.width() + x] = max_val;
        }
    }
    out
}

/// Opening = erosion then dilation (removes small bright spots).
pub fn open(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    dilate(&erode(img, se), se)
}

/// Closing = dilation then erosion (removes small dark spots).
pub fn close(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    erode(&dilate(img, se), se)
}

/// Morphological gradient = dilation - erosion (edge outline).
pub fn gradient(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let d = dilate(img, se);
    let e = erode(img, se);
    d.zip_with(&e, |a, b| a - b).unwrap_or_else(|_| img.clone())
}

/// Top hat = image - opening (reveals bright features smaller than SE).
pub fn top_hat(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let opened = open(img, se);
    img.zip_with(&opened, |a, b| a - b).unwrap_or_else(|_| img.clone())
}

/// Black hat = closing - image (reveals dark features smaller than SE).
pub fn black_hat(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let closed = close(img, se);
    closed.zip_with(img, |a, b| a - b).unwrap_or_else(|_| img.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> GrayImage {
        let mut img = GrayImage::new(11, 11);
        // White square in center
        for y in 3..8 {
            for x in 3..8 {
                img.set(x, y, 255.0).unwrap();
            }
        }
        img
    }

    #[test]
    fn test_erosion_shrinks_object() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let eroded = erode(&img, &se);
        // Corner of original white square should be eroded (0)
        assert_eq!(eroded.get(3, 3).unwrap(), 0.0);
        // Interior should remain white
        assert_eq!(eroded.get(5, 5).unwrap(), 255.0);
    }

    #[test]
    fn test_dilation_grows_object() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let dilated = dilate(&img, &se);
        // Pixel just outside original should be dilated
        assert_eq!(dilated.get(2, 5).unwrap(), 255.0);
        // Pixel well outside should remain 0
        assert_eq!(dilated.get(0, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_opening_removes_noise() {
        let mut img = GrayImage::new(11, 11);
        // Large object
        for y in 3..8 {
            for x in 3..8 {
                img.set(x, y, 200.0).unwrap();
            }
        }
        // Small bright speck
        img.set(0, 0, 255.0).unwrap();

        let se = StructuringElement::square(3);
        let opened = open(&img, &se);
        // Speck removed
        assert!(opened.get(0, 0).unwrap() < 50.0);
        // Large object mostly preserved
        assert!(opened.get(5, 5).unwrap() > 150.0);
    }

    #[test]
    fn test_closing_fills_gaps() {
        let mut img = GrayImage::new(11, 11);
        // Large bright region with small dark hole
        for y in 2..9 {
            for x in 2..9 {
                img.set(x, y, 200.0).unwrap();
            }
        }
        img.set(5, 5, 0.0).unwrap(); // hole

        let se = StructuringElement::square(3);
        let closed = close(&img, &se);
        // Hole should be filled
        assert!(closed.get(5, 5).unwrap() > 0.0);
    }

    #[test]
    fn test_gradient_produces_outline() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let grad = gradient(&img, &se);
        // Interior should have low gradient
        assert!(grad.get(5, 5).unwrap() < 10.0);
        // Edge should have high gradient
        assert!(grad.get(3, 5).unwrap() > 100.0);
    }

    #[test]
    fn test_cross_structuring_element() {
        let se = StructuringElement::cross(5);
        // Center row all true
        for x in 0..5 {
            assert!(se.get(x, 2));
        }
        // Center col all true
        for y in 0..5 {
            assert!(se.get(2, y));
        }
        // Corner should be false
        assert!(!se.get(0, 0));
    }

    #[test]
    fn test_top_hat() {
        let mut img = GrayImage::new(15, 15);
        // Uniform background
        for y in 0..15 {
            for x in 0..15 {
                img.set(x, y, 100.0).unwrap();
            }
        }
        // Small bright bump
        img.set(7, 7, 200.0).unwrap();

        let se = StructuringElement::square(3);
        let th = top_hat(&img, &se);
        // The bump should be highlighted
        assert!(th.get(7, 7).unwrap() > 50.0);
    }

    #[test]
    fn test_black_hat() {
        let mut img = GrayImage::new(15, 15);
        for y in 0..15 {
            for x in 0..15 {
                img.set(x, y, 100.0).unwrap();
            }
        }
        // Small dark spot
        img.set(7, 7, 20.0).unwrap();

        let se = StructuringElement::square(3);
        let bh = black_hat(&img, &se);
        assert!(bh.get(7, 7).unwrap() > 30.0);
    }
}
