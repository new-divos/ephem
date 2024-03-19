use std::{marker::PhantomData, iter::{IntoIterator, Iterator}, ops::{Index, IndexMut}};

use crate::core::vectors::{Cartesian, Vec3d};

/// Represents a 3x3 matrix.
pub struct Mat3d([f64; 9]);

impl Mat3d {
    /// Constructs a new matrix filled with zeros.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self([0.0; 9])
    }

    /// Constructs a new matrix filled with ones.
    #[inline(always)]
    pub fn ones() -> Self {
        Self([1.0; 9])
    }

    /// Constructs a new identity matrix.
    ///
    /// The identity matrix is a square matrix with ones on the main diagonal
    /// and zeros elsewhere.
    #[inline(always)]
    pub fn eye() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Constructs a rotation matrix around the x-axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - The rotation angle in radians.
    pub fn r_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c])
    }

    /// Constructs a rotation matrix around the y-axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - The rotation angle in radians.
    pub fn r_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c])
    }

    /// Constructs a rotation matrix around the z-axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - The rotation angle in radians.
    pub fn f_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])
    }

    /// Constructs a matrix from row vectors.
    ///
    /// Constructs a matrix using three row vectors. The row vectors represent
    /// the rows of the resulting matrix.
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

    /// Constructs a matrix from column vectors.
    ///
    /// Constructs a matrix using three column vectors. The column vectors represent
    /// the columns of the resulting matrix.
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

    /// Returns a view of the matrix as rows.
    ///
    /// This method constructs and returns a `Mat3dView`, which provides a view of the matrix where each row is represented by a `Vec3d<Cartesian>` vector.
    ///
    /// # Returns
    ///
    /// A `Mat3dView` containing three `Vec3d<Cartesian>` vectors, each representing a row of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::{
    ///     vectors::{Cartesian, CartesianBuilder, Vec3d},
    ///     matrices::Mat3d,
    /// };
    /// 
    /// let row1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let row2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// let row3 = CartesianBuilder::with(7.0, 8.0, 9.0).build();
    ///
    /// let mat = Mat3d::with_rows(row1, row2, row3);
    /// let rows = mat.rows();
    ///
    /// assert_eq!(rows[0].x(), 1.0);
    /// assert_eq!(rows[1].y(), 5.0);
    /// assert_eq!(rows[2].z(), 9.0);
    /// ```
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

    /// Returns a view of the matrix as columns.
    ///
    /// This method constructs and returns a `Mat3dView`, which provides a view of the matrix where each column is represented by a `Vec3d<Cartesian>` vector.
    ///
    /// # Returns
    ///
    /// A `Mat3dView` containing three `Vec3d<Cartesian>` vectors, each representing a column of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::{
    ///     vectors::{Cartesian, CartesianBuilder, Vec3d},
    ///     matrices::Mat3d,
    /// };
    /// 
    /// let col1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let col2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// let col3 = CartesianBuilder::with(7.0, 8.0, 9.0).build();
    ///
    /// let mat = Mat3d::with_columns(col1, col2, col3);
    /// let columns = mat.columns();
    ///
    /// assert_eq!(columns[0].x(), 1.0);
    /// assert_eq!(columns[1].y(), 5.0);
    /// assert_eq!(columns[2].z(), 9.0);
    /// ```
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

    #[inline(always)]
    pub fn iter(&self) -> Mat3dIter<'_> {
        Mat3dIter {
            data: &self.0,
            cursor: 0,
        }
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> Mat3dMutIter<'_> {
        Mat3dMutIter {
            data: &mut self.0,
            cursor: 0,
        }
    }
}

impl AsRef<[f64]> for Mat3d {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

impl AsMut<[f64]> for Mat3d {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

/// Implements indexing for accessing elements of the `Mat3d` matrix using `(row, column)` tuples.
///
/// This allows accessing individual elements of the matrix using zero-based row and column indices.
/// The returned reference points to the element at the specified row and column.
impl Index<(usize, usize)> for Mat3d {
    type Output = f64;

    /// Returns a reference to the element at the specified `(row, column)` indices.
    ///
    /// # Arguments
    ///
    /// * `index` - A tuple representing the zero-based row and column indices.
    ///
    /// # Returns
    ///
    /// A reference to the element at the specified row and column.
    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[3 * index.0 + index.1]
    }
}

/// Implements mutable indexing for accessing and modifying elements of the `Mat3d` matrix using `(row, column)` tuples.
///
/// This allows accessing and modifying individual elements of the matrix using zero-based row and column indices.
/// The returned mutable reference allows modification of the element at the specified row and column.
impl IndexMut<(usize, usize)> for Mat3d {
    /// Returns a mutable reference to the element at the specified `(row, column)` indices.
    ///
    /// # Arguments
    ///
    /// * `index` - A tuple representing the zero-based row and column indices.
    ///
    /// # Returns
    ///
    /// A mutable reference to the element at the specified row and column.
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[3 * index.0 + index.1]
    }
}

impl<'a> IntoIterator for &'a Mat3d {
    type Item = f64;
    type IntoIter = Mat3dIter<'a>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Mat3d {
    type Item = &'a mut f64;
    type IntoIter = Mat3dMutIter<'a>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl IntoIterator for Mat3d {
    type Item = f64;
    type IntoIter = Mat3dIntoIter;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Mat3dIntoIter {
            data: self.0,
            cursor: 0,
        }
    }
}

pub struct Mat3dIter<'a> {
    data: &'a [f64],
    cursor: usize,
}

impl<'a> Iterator for Mat3dIter<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.data.len() {
            let value = self.data[self.cursor];
            self.cursor += 1;

            Some(value)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item>
        where
            Self: Sized, {
        Some(self.data[self.data.len() - 1])
    }
}

pub struct Mat3dMutIter<'a> {
    data: &'a mut [f64],
    cursor: usize,
}

impl<'a> Iterator for Mat3dMutIter<'a> {
    type Item = &'a mut f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.data.len() {
            let i = self.cursor;
            self.cursor += 1;

            let data_ptr = self.data.as_mut_ptr();
            unsafe { Some(&mut *data_ptr.add(i)) }
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
        where
            Self: Sized, {
        let data_ptr = self.data.as_mut_ptr();
        unsafe { Some(&mut *data_ptr.add(self.data.len() - 1)) }
    }
}

pub struct Mat3dIntoIter {
    data: [f64; 9],
    cursor: usize,
}

impl Iterator for Mat3dIntoIter {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.data.len() {
            let value = self.data[self.cursor];
            self.cursor += 1;

            Some(value)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item>
        where
            Self: Sized, {
        Some(self.data[self.data.len() - 1])
    }
}

/// Represents a view into a 3x3 matrix (`Mat3d`) consisting of three Cartesian vectors.
///
/// This structure provides methods and implementations to interact with the matrix view.
pub struct Mat3dView([Vec3d<Cartesian>; 3]);

impl Mat3dView {
    /// Returns an iterator over the rows of the matrix view.
    ///
    /// This method constructs and returns an iterator over the rows of the matrix view,
    /// allowing iteration through each row.
    #[inline(always)]
    pub fn iter(&self) -> Mat3dViewIter<'_> {
        Mat3dViewIter {
            data: &self.0,
            cursor: 0,
        }
    }
}

/// Implements the trait for treating `Mat3dView` as a reference to an array of `Vec3d<Cartesian>` vectors.
///
/// This implementation allows treating `Mat3dView` as a reference to an array of Cartesian vectors,
/// enabling operations and methods that accept slice references.
impl AsRef<[Vec3d<Cartesian>]> for Mat3dView {
    /// Returns a reference to the underlying array of Cartesian vectors.
    ///
    /// # Returns
    ///
    /// A reference to the array of Cartesian vectors contained within the `Mat3dView`.
    #[inline(always)]
    fn as_ref(&self) -> &[Vec3d<Cartesian>] {
        &self.0
    }
}

/// Implements indexing for accessing individual rows of the `Mat3dView` matrix.
///
/// This implementation enables accessing individual rows of the matrix view using zero-based indexing.
impl Index<usize> for Mat3dView {
    type Output = Vec3d<Cartesian>;

    /// Returns a reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The zero-based index indicating the row to access.
    ///
    /// # Returns
    ///
    /// A reference to the Cartesian vector representing the row at the specified index.
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Implements conversion into an iterator for borrowing a reference to a `Mat3dView`, yielding shared references to `Vec3d<Cartesian>` vectors.
///
/// This implementation converts a reference to a `Mat3dView` into an iterator, allowing iteration over its rows while retaining ownership of the view.
/// It yields shared references to `Vec3d<Cartesian>` vectors as it iterates through each row of the matrix view.
impl<'a> IntoIterator for &'a Mat3dView {
    /// The type of the items yielded by the iterator, which is a shared reference to `Vec3d<Cartesian>` vector.
    type Item = &'a Vec3d<Cartesian>;
    /// The type of the iterator produced by the conversion, which is a `Mat3dViewIter`.
    type IntoIter = Mat3dViewIter<'a>;

    /// Borrows the `Mat3dView` and returns an iterator over its rows.
    ///
    /// Each row of the matrix view is yielded as a shared reference to a `Vec3d<Cartesian>` vector.
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Implements conversion into an iterator for consuming a `Mat3dView`, yielding `Vec3d<Cartesian>` vectors.
///
/// This implementation converts a `Mat3dView` into an iterator, consuming the view in the process.
/// It yields `Vec3d<Cartesian>` vectors as it iterates through each row of the matrix view.
impl IntoIterator for Mat3dView {
    /// The type of the items yielded by the iterator, which is a `Vec3d<Cartesian>` vector.
    type Item = Vec3d<Cartesian>;
    /// The type of the iterator produced by the conversion, which is a `Mat3dViewIntoIter`.
    type IntoIter = Mat3dViewIntoIter;

    /// Consumes the `Mat3dView`, returning an iterator over its rows.
    ///
    /// Each row of the matrix view is yielded as a `Vec3d<Cartesian>` vector.
    fn into_iter(self) -> Self::IntoIter {
        // Reverse the rows to maintain the correct order during iteration
        let mut data = self.0;
        data.reverse();

        Mat3dViewIntoIter(data.into_iter().collect())
    }
}

/// Iterator over the rows of a `Mat3dView`.
///
/// This iterator allows iterating over the rows of a `Mat3dView`, providing references to each row's `Vec3d<Cartesian>` vector.
pub struct Mat3dViewIter<'a> {
    /// Reference to the underlying array of `Vec3d<Cartesian>` vectors.
    data: &'a [Vec3d<Cartesian>],
    /// Cursor indicating the current position within the array.
    cursor: usize,
}

impl<'a> Iterator for Mat3dViewIter<'a> {
    /// The type of the items yielded by the iterator, which is a reference to a `Vec3d<Cartesian>` vector.
    type Item = &'a Vec3d<Cartesian>;

    /// Advances the iterator and returns the next row of the matrix view.
    ///
    /// Returns `Some(reference)` to the next row if available, otherwise `None`.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.data.len() {
            let i = self.cursor;
            self.cursor += 1;

            Some(&self.data[i])
        } else {
            None
        }
    }
}

/// Iterator that consumes a `Mat3dView`, yielding `Vec3d<Cartesian>` vectors.
///
/// This iterator consumes a `Mat3dView`, yielding `Vec3d<Cartesian>` vectors as it iterates through each row of the matrix view.
pub struct Mat3dViewIntoIter(Vec<Vec3d<Cartesian>>);

impl Iterator for Mat3dViewIntoIter {
    /// The type of the items yielded by the iterator, which is a `Vec3d<Cartesian>` vector.
    type Item = Vec3d<Cartesian>;

    /// Advances the iterator and returns the next `Vec3d<Cartesian>` vector.
    ///
    /// Returns `Some(vector)` if a vector is available, otherwise `None`.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.0.is_empty() {
            self.0.pop()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test function for Mat3d creation methods.
    ///
    /// This test function verifies the correctness of the `zeros()`, `ones()`, and `eye()` methods of the `Mat3d` struct.
    /// It iterates over each element of the matrices created by these methods and asserts that they have the expected values.
    #[test]
    fn mat3d_creation_test() {
        // Test zeros matrix creation
        let z = Mat3d::zeros();
        for value in z.0 {
            assert_eq!(value, 0.0);
        }

        // Test ones matrix creation
        let o = Mat3d::ones();
        for value in o.0 {
            assert_eq!(value, 1.0);
        }

        // Test identity matrix creation
        let e = Mat3d::eye();
        for (i, value) in e.0.iter().enumerate() {
            assert_eq!(*value, if i / 3 == i % 3 { 1.0 } else { 0.0 });
        }
    }
}
