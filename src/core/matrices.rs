use std::{marker::PhantomData, iter::{IntoIterator, Iterator}, ops::{Index, IndexMut}};

use crate::core::vectors::{Cartesian, Vec3d};

pub struct Mat3d([f64; 9]);

impl Mat3d {
    #[inline]
    pub fn zeros() -> Self {
        Self([0.0; 9])
    }

    #[inline]
    pub fn ones() -> Self {
        Self([1.0; 9])
    }

    #[inline]
    pub fn eye() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    pub fn r_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c])
    }

    pub fn r_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c])
    }

    pub fn f_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])
    }

    #[inline]
    #[rustfmt::skip]
    pub fn with_rows(
        row1: Vec3d<Cartesian>,
        row2: Vec3d<Cartesian>,
        row3: Vec3d<Cartesian>,
    ) -> Self {
        Self([
            row1.0[0], row1.0[1], row1.0[2], 
            row2.0[0], row2.0[1], row2.0[2], 
            row3.0[0], row3.0[1], row3.0[2],
        ])
    }

    #[inline]
    #[rustfmt::skip]
    pub fn with_columns(
        col1: Vec3d<Cartesian>,
        col2: Vec3d<Cartesian>,
        col3: Vec3d<Cartesian>
    ) -> Self {
        Self([
            col1.0[0], col2.0[0], col3.0[0],
            col1.0[1], col2.0[1], col3.0[1],
            col1.0[2], col2.0[2], col3.0[2]
        ])
    }

    #[inline]
    pub fn rows(&self) -> Mat3dView {
        Mat3dView(
            [
                Vec3d::<Cartesian> (
                    [self.0[0], self.0[1], self.0[2]], 
                    PhantomData::<Cartesian> {}
                ),
                Vec3d::<Cartesian> (
                    [self.0[3], self.0[4], self.0[5]], 
                    PhantomData::<Cartesian> {}
                ),
                Vec3d::<Cartesian> (
                    [self.0[6], self.0[7], self.0[8]], 
                    PhantomData::<Cartesian> {}
                )
            ],
        )
    }

    #[inline]
    pub fn columns(&self) -> Mat3dView {
        Mat3dView(
            [
                Vec3d::<Cartesian> (
                    [self.0[0], self.0[3], self.0[6]], 
                    PhantomData::<Cartesian> {}
                ),
                Vec3d::<Cartesian> (
                    [self.0[1], self.0[4], self.0[7]], 
                    PhantomData::<Cartesian> {}
                ),
                Vec3d::<Cartesian> (
                    [self.0[2], self.0[5], self.0[8]], 
                    PhantomData::<Cartesian> {}
                )
            ],
        )
    }
}

impl Index<(usize, usize)> for Mat3d {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let index = 3 * row + col;

        &self.0[index]
    }
}

impl IndexMut<(usize, usize)> for Mat3d {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let index = 3 * row + col;

        &mut self.0[index]
    }
}

pub struct Mat3dView([Vec3d<Cartesian>; 3]);

impl Mat3dView {
    #[inline]
    pub fn iter(&self) -> Mat3dViewIter<'_> {
        Mat3dViewIter {
            data: &self.0,
            cursor: 0,
        }
    }
}

impl AsRef<[Vec3d<Cartesian>]> for Mat3dView {
    #[inline]
    fn as_ref(&self) -> &[Vec3d<Cartesian>] {
        &self.0
    }
}

impl Index<usize> for Mat3dView {
    type Output = Vec3d<Cartesian>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IntoIterator for Mat3dView {
    type Item = Vec3d<Cartesian>;
    type IntoIter = Mat3dViewIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let mut data = self.0;
        data.reverse();

        Mat3dViewIntoIter {
            data: data.into_iter().collect(),
            cursor: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Mat3dView {
    type Item = &'a Vec3d<Cartesian>;
    type IntoIter = Mat3dViewIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Mat3dViewIter<'a> {
    data: &'a [Vec3d<Cartesian>],
    cursor: usize,
}

impl<'a> Iterator for Mat3dViewIter<'a> {
    type Item = &'a Vec3d<Cartesian>;

    fn next(&mut self) -> Option<Self::Item> {
        if (0..3).contains(&self.cursor) {
            let i = self.cursor;
            self.cursor += 1;

            Some(&self.data[i])
        } else {
            None
        }
    }
}

pub struct Mat3dViewIntoIter {
    data: Vec<Vec3d<Cartesian>>,
    cursor: usize
}

impl Iterator for Mat3dViewIntoIter {
    type Item = Vec3d<Cartesian>;

    fn next(&mut self) -> Option<Self::Item> {
        if (0..3).contains(&self.cursor) {
            self.cursor += 1;
            self.data.pop()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat3d_creation_test() {
        let z = Mat3d::zeros();
        for value in z.0 {
            assert_eq!(value, 0.0);
        }

        let o = Mat3d::ones();
        for value in o.0 {
            assert_eq!(value, 1.0);
        }

        let e = Mat3d::eye();
        for (i, value) in e.0.iter().enumerate() {
            assert_eq!(*value, if i / 3 == i % 3 { 1.0 } else { 0.0 });
        }
    }
}
