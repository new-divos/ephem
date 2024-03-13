use std::{marker::PhantomData, iter::Iterator, ops::{Index, IndexMut}};

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
    pub fn row(&self, index: usize) -> Option<Vec3d<Cartesian>> {
        if index < 3 {
            let idx3 = index * 3;
            Some(
                Vec3d::<Cartesian>(
                    [self.0[idx3], self.0[idx3 + 1], self.0[idx3 + 2]],
                    PhantomData::<Cartesian> {}
                )
            )
        } else {
            None
        }
    }

    #[inline]
    pub fn iter_rows(&self) -> Mat3dRowsIter<'_> {
        Mat3dRowsIter {
            data: &self.0,
            cursor: 0,
        }
    }

    #[inline]
    pub fn column(&self, index: usize) -> Option<Vec3d<Cartesian>> {
        if index < 3 {
            Some(
                Vec3d::<Cartesian>(
                    [self.0[index], self.0[3 + index], self.0[6 + index]],
                    PhantomData::<Cartesian> {}
                )
            )
        } else {
            None
        }
    }

    #[inline]
    pub fn iter_columns(&self) -> Mat3dColumnsIter<'_> {
        Mat3dColumnsIter {
            data: &self.0,
            cursor: 0,
        }
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

pub struct Mat3dRowsIter<'a> {
    data: &'a [f64],
    cursor: usize,
}

impl<'a> Iterator for Mat3dRowsIter<'a> {
    type Item = Vec3d<Cartesian>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < 3 {
            let idx3 = 3 * self.cursor;
            let vector = Vec3d::<Cartesian>(
                [self.data[idx3], self.data[idx3 + 1], self.data[idx3 + 2]],
                PhantomData::<Cartesian> {}
            );
            self.cursor += 1;

            Some(vector)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (3, Some(3))
    }
}

pub struct Mat3dColumnsIter<'a> {
    data: &'a [f64],
    cursor: usize,
}

impl<'a> Iterator for Mat3dColumnsIter<'a> {
    type Item = Vec3d<Cartesian>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < 3 {
            let vector = Vec3d::<Cartesian>(
                [self.data[self.cursor], self.data[3 + self.cursor], self.data[6 + self.cursor]],
                PhantomData::<Cartesian> {}
            );
            self.cursor += 1;

            Some(vector)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (3, Some(3))
    }
}