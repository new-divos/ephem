use std::{
    iter::{FromIterator, IntoIterator, Iterator},
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::core::vectors::{Cartesian, Vec3d};

/// Represents a 3x3 matrix.
#[derive(Debug)]
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
    #[rustfmt::skip]
    #[inline]
    pub fn with_rows(
        row1: Vec3d<Cartesian>,
        row2: Vec3d<Cartesian>,
        row3: Vec3d<Cartesian>,
    ) -> Self {
        Self([
            row1.0[0], row1.0[1], row1.0[2],
            row2.0[0], row2.0[1], row2.0[2],
            row3.0[0], row3.0[1], row3.0[2]
        ])
    }

    /// Constructs a matrix from column vectors.
    ///
    /// Constructs a matrix using three column vectors. The column vectors represent
    /// the columns of the resulting matrix.
    #[rustfmt::skip]
    #[inline]
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
        Mat3dView([
            Vec3d::<Cartesian>(
                [self.0[0], self.0[1], self.0[2]],
                PhantomData::<Cartesian> {},
            ),
            Vec3d::<Cartesian>(
                [self.0[3], self.0[4], self.0[5]],
                PhantomData::<Cartesian> {},
            ),
            Vec3d::<Cartesian>(
                [self.0[6], self.0[7], self.0[8]],
                PhantomData::<Cartesian> {},
            ),
        ])
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
        Mat3dView([
            Vec3d::<Cartesian>(
                [self.0[0], self.0[3], self.0[6]],
                PhantomData::<Cartesian> {},
            ),
            Vec3d::<Cartesian>(
                [self.0[1], self.0[4], self.0[7]],
                PhantomData::<Cartesian> {},
            ),
            Vec3d::<Cartesian>(
                [self.0[2], self.0[5], self.0[8]],
                PhantomData::<Cartesian> {},
            ),
        ])
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

/// Implements the `AsRef` trait for `Mat3d`.
///
/// This trait allows obtaining a reference to the inner array of `f64` values
/// stored in a `Mat3d` instance. The reference returned by `as_ref` provides
/// access to the raw data of the `Mat3d` in the form of a slice.
impl AsRef<[f64]> for Mat3d {
    /// `as_ref` returns a reference to the underlying array of `f64` values. This
    /// can be useful when interoperating with functions or libraries that expect
    /// a slice of `f64` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mat = Mat3d::eye();
    /// let data_ref: &[f64] = mat.as_ref();
    /// assert_eq!(data_ref.len(), 9);
    /// ```
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

/// Implements the `AsMut` trait for `Mat3d`.
///
/// This trait allows obtaining a mutable reference to the inner array of `f64` values
/// stored in a `Mat3d` instance. The reference returned by `as_mut` provides
/// mutable access to the raw data of the `Mat3d` in the form of a slice.
impl AsMut<[f64]> for Mat3d {
    /// `as_mut` returns a mutable reference to the underlying array of `f64` values. This
    /// can be useful when modifying the contents of the `Mat3d` directly through the
    /// raw data array.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mut mat = Mat3d::eye();
    /// {
    ///     let data_ref: &mut [f64] = mat.as_mut();
    ///     data_ref[0] = 2.0;
    /// }
    /// assert_eq!(mat[(0, 0)], 2.0);
    /// ```
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

/// Implements the `Clone` trait for `Mat3d`.
///
/// This trait allows creating a deep copy of a `Mat3d` instance, including its inner data.
impl Clone for Mat3d {
    /// The `clone` method performs a deep copy of the `Mat3d` instance, creating a new instance
    /// with the same values as the original. This includes copying the underlying array of `f64` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mat1 = Mat3d::eye();
    /// let mat2 = mat1.clone();
    ///
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         assert_eq!(mat1[(i, j)], mat2[(i, j)]);
    ///     }
    /// }
    /// ```
    #[rustfmt::skip]
    #[inline(always)]
    fn clone(&self) -> Self {
        Self([
            self.0[0], self.0[1], self.0[2],
            self.0[3], self.0[4], self.0[5],
            self.0[6], self.0[7], self.0[8]
        ])
    }
}

/// Implements the `PartialEq` trait for `Mat3d`.
///
/// This trait allows comparing two `Mat3d` instances for equality.
impl PartialEq for Mat3d {
    /// The `eq` method checks if each corresponding element in the two matrices are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mat1 = Mat3d::eye();
    /// let mat2 = Mat3d::eye();
    ///
    /// assert_eq!(mat1, mat2); // Both matrices are equal
    /// ```
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0[0].eq(&other.0[0])
            && self.0[1].eq(&other.0[1])
            && self.0[2].eq(&other.0[2])
            && self.0[3].eq(&other.0[3])
            && self.0[4].eq(&other.0[4])
            && self.0[5].eq(&other.0[5])
            && self.0[6].eq(&other.0[6])
            && self.0[7].eq(&other.0[7])
            && self.0[8].eq(&other.0[8])
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

/// Implements the `FromIterator` trait for `Mat3d`.
///
/// This trait allows creating a `Mat3d` from an iterator over elements convertible to `f64`.
impl<R> FromIterator<R> for Mat3d
where
    f64: From<R>,
{
    /// The `from_iter` method consumes the provided iterator and constructs a `Mat3d` using its elements.
    /// If the iterator produces less than 9 elements, the remaining elements are initialized with zeros.
    /// If the iterator produces more than 9 elements, the excess elements are ignored.
    ///
    /// # Type Parameters
    ///
    /// - `R`: The type of elements convertible to `f64`.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator over elements convertible to `f64` values.
    ///
    /// # Returns
    ///
    /// A matrix 3x3 constructed from the first nine values encountered in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::iter::FromIterator;
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    /// let mat: Mat3d = data.into_iter().collect();
    ///
    /// assert_eq!(mat[(0, 0)], 1.0);
    /// assert_eq!(mat[(1, 1)], 5.0);
    /// assert_eq!(mat[(2, 2)], 9.0);
    /// ```
    fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
        let mut arr = [0.0f64; 9];
        for (idx, value) in iter.into_iter().enumerate().take(arr.len()) {
            arr[idx] = value.into();
        }

        Self(arr)
    }
}

/// Implements the unary negation operation for a reference to `Mat3d`.
impl<'a> Neg for &'a Mat3d {
    type Output = Mat3d;

    /// The `neg` method negates each element of the matrix and returns a new `Mat3d` with the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mat = Mat3d::eye();
    /// let neg_mat = -&mat;
    ///
    /// assert_eq!(neg_mat[(0, 0)], -1.0);
    /// assert_eq!(neg_mat[(1, 1)], -1.0);
    /// assert_eq!(neg_mat[(2, 2)], -1.0);
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn neg(self) -> Self::Output {
        Mat3d([
            -self.0[0], -self.0[1], -self.0[2],
            -self.0[3], -self.0[4], -self.0[5],
            -self.0[6], -self.0[7], -self.0[8],
        ])
    }
}

/// Implements the unary negation operation for `Mat3d`.
impl Neg for Mat3d {
    type Output = Mat3d;

    /// The `neg` method negates each element of the matrix and returns a new `Mat3d` with the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mat = Mat3d::eye();
    /// let neg_mat = -mat;
    ///
    /// assert_eq!(neg_mat[(0, 0)], -1.0);
    /// assert_eq!(neg_mat[(1, 1)], -1.0);
    /// assert_eq!(neg_mat[(2, 2)], -1.0);
    /// ```
    #[inline(always)]
    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}

/// Implements the addition operation for two references to `Mat3d`, returning a new `Mat3d`.
impl<'a> Add for &'a Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise addition of the corresponding elements of the two matrices.
    /// The resulting matrix contains the sum of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = &m1 + &m2;
    ///
    /// assert_eq!(result[(0, 0)], 2.0);
    /// assert_eq!(result[(1, 1)], 2.0);
    /// assert_eq!(result[(2, 2)], 2.0);
    /// ```
    /// 
    /// # Panics
    /// 
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Mat3d([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
            self.0[8] + rhs.0[8],
        ])
    }
}

/// Implements addition between a reference to a `Mat3d` matrix and an owned `Mat3d` matrix, returning a new `Mat3d`.
impl<'a> Add<Mat3d> for &'a Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise addition of the corresponding elements of the two matrices.
    /// The resulting matrix contains the sum of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = &m1 + m2;
    ///
    /// assert_eq!(result[(0, 0)], 2.0);
    /// assert_eq!(result[(1, 1)], 2.0);
    /// assert_eq!(result[(2, 2)], 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn add(self, rhs: Mat3d) -> Self::Output {
        self.add(&rhs)
    }
}

/// Implements addition between an owned `Mat3d` matrix and a reference to a `Mat3d` matrix, returning a new `Mat3d`.
impl<'a> Add<&'a Mat3d> for Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise addition of the corresponding elements of the two matrices.
    /// The resulting matrix contains the sum of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = m1 + &m2;
    ///
    /// assert_eq!(result[(0, 0)], 2.0);
    /// assert_eq!(result[(1, 1)], 2.0);
    /// assert_eq!(result[(2, 2)], 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn add(self, rhs: &'a Mat3d) -> Self::Output {
        (&self).add(rhs)
    }
}

/// Implements addition between two owned `Mat3d` matrices, returning a new `Mat3d`.
impl Add for Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise addition of the corresponding elements of the two matrices.
    /// The resulting matrix contains the sum of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = m1 + m2;
    ///
    /// assert_eq!(result[(0, 0)], 2.0);
    /// assert_eq!(result[(1, 1)], 2.0);
    /// assert_eq!(result[(2, 2)], 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

/// Implements in-place addition between a mutable `Mat3d` matrix and an immutable reference to another `Mat3d`.
impl<'a> AddAssign<&'a Mat3d> for Mat3d {
    /// This function performs element-wise addition of the corresponding elements of the two matrices
    /// and assigns the result to the mutable matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mut m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// m1 += &m2;
    ///
    /// assert_eq!(m1[(0, 0)], 2.0);
    /// assert_eq!(m1[(1, 1)], 2.0);
    /// assert_eq!(m1[(2, 2)], 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline]
    fn add_assign(&mut self, rhs: &'a Mat3d) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
        self.0[4] += rhs.0[4];
        self.0[5] += rhs.0[5];
        self.0[6] += rhs.0[6];
        self.0[7] += rhs.0[7];
        self.0[8] += rhs.0[8];
    }
}

/// Implements in-place addition between two `Mat3d` matrices.
impl AddAssign for Mat3d {
    /// This function performs element-wise addition of the corresponding elements of the two matrices
    /// and assigns the result to the mutable matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mut m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// m1 += m2;
    ///
    /// assert_eq!(m1[(0, 0)], 2.0);
    /// assert_eq!(m1[(1, 1)], 2.0);
    /// assert_eq!(m1[(2, 2)], 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

/// Implements the substraction operation for two references to `Mat3d`, returning a new `Mat3d`.
impl<'a> Sub for &'a Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise substraction of the corresponding elements of the two matrices.
    /// The resulting matrix contains the difference of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = &m1 - &m2;
    ///
    /// assert_eq!(result[(0, 0)], 0.0);
    /// assert_eq!(result[(1, 1)], 0.0);
    /// assert_eq!(result[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    fn sub(self, rhs: Self) -> Self::Output {
        Mat3d([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6],
            self.0[7] - rhs.0[7],
            self.0[8] - rhs.0[8],
        ])
    }
}

/// Implements substraction between a reference to a `Mat3d` matrix and an owned `Mat3d` matrix, returning a new `Mat3d`.
impl<'a> Sub<Mat3d> for &'a Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise substraction of the corresponding elements of the two matrices.
    /// The resulting matrix contains the difference of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = &m1 - m2;
    ///
    /// assert_eq!(result[(0, 0)], 0.0);
    /// assert_eq!(result[(1, 1)], 0.0);
    /// assert_eq!(result[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn sub(self, rhs: Mat3d) -> Self::Output {
        self.sub(&rhs)
    }
}

/// Implements substraction between an owned `Mat3d` matrix and a reference to a `Mat3d` matrix, returning a new `Mat3d`.
impl<'a> Sub<&'a Mat3d> for Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise substraction of the corresponding elements of the two matrices.
    /// The resulting matrix contains the difference of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = m1 - &m2;
    ///
    /// assert_eq!(result[(0, 0)], 0.0);
    /// assert_eq!(result[(1, 1)], 0.0);
    /// assert_eq!(result[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn sub(self, rhs: &'a Mat3d) -> Self::Output {
        (&self).sub(rhs)
    }
}

/// Implements addition between two owned `Mat3d` matrices, returning a new `Mat3d`.
impl Sub for Mat3d {
    type Output = Mat3d;

    /// This function performs element-wise substraction of the corresponding elements of the two matrices.
    /// The resulting matrix contains the difference of each corresponding pair of elements from the input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// let result = m1 - m2;
    ///
    /// assert_eq!(result[(0, 0)], 0.0);
    /// assert_eq!(result[(1, 1)], 0.0);
    /// assert_eq!(result[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

/// Implements in-place substraction between a mutable `Mat3d` matrix and an immutable reference to another `Mat3d`.
impl<'a> SubAssign<&'a Mat3d> for Mat3d {
    /// This function performs element-wise substraction of the corresponding elements of the two matrices
    /// and assigns the result to the mutable matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mut m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// m1 -= &m2;
    ///
    /// assert_eq!(m1[(0, 0)], 0.0);
    /// assert_eq!(m1[(1, 1)], 0.0);
    /// assert_eq!(m1[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Mat3d) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
        self.0[3] -= rhs.0[3];
        self.0[4] -= rhs.0[4];
        self.0[5] -= rhs.0[5];
        self.0[6] -= rhs.0[6];
        self.0[7] -= rhs.0[7];
        self.0[8] -= rhs.0[8];
    }
}

/// Implements in-place substraction between two `Mat3d` matrices.
impl SubAssign for Mat3d {
    /// This function performs element-wise substraction of the corresponding elements of the two matrices
    /// and assigns the result to the mutable matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::matrices::Mat3d;
    ///
    /// let mut m1 = Mat3d::eye();
    /// let m2 = Mat3d::ones();
    ///
    /// m1 -= m2;
    ///
    /// assert_eq!(m1[(0, 0)], 0.0);
    /// assert_eq!(m1[(1, 1)], 0.0);
    /// assert_eq!(m1[(2, 2)], 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if any of the elements in the resulting matrix is NaN.
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<'a> Mul<f64> for &'a Mat3d {
    type Output = Mat3d;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Mat3d([
            rhs * self.0[0],
            rhs * self.0[1],
            rhs * self.0[2],
            rhs * self.0[3],
            rhs * self.0[4],
            rhs * self.0[5],
            rhs * self.0[6],
            rhs * self.0[7],
            rhs * self.0[8],
        ])
    }
}

impl<'a> Mul<&'a Mat3d> for f64 {
    type Output = Mat3d;

    #[inline(always)]
    fn mul(self, rhs: &'a Mat3d) -> Self::Output {
        rhs.mul(self)
    }
}

impl Mul<f64> for Mat3d {
    type Output = Mat3d;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl Mul<Mat3d> for f64 {
    type Output = Mat3d;

    #[inline(always)]
    fn mul(self, rhs: Mat3d) -> Self::Output {
        (&rhs).mul(self)
    }
}

impl MulAssign<f64> for Mat3d {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
        self.0[4] *= rhs;
        self.0[5] *= rhs;
        self.0[6] *= rhs;
        self.0[7] *= rhs;
        self.0[8] *= rhs;
    }
}

impl<'a> Mul<Vec3d<Cartesian>> for &'a Mat3d {
    type Output = Vec3d<Cartesian>;

    #[inline]
    fn mul(self, rhs: Vec3d<Cartesian>) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1] + self.0[2] * rhs.0[2],
                self.0[3] * rhs.0[0] + self.0[4] * rhs.0[1] + self.0[5] * rhs.0[2],
                self.0[6] * rhs.0[0] + self.0[7] * rhs.0[1] + self.0[8] * rhs.0[2],
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

impl Mul<Vec3d<Cartesian>> for Mat3d {
    type Output = Vec3d<Cartesian>;

    #[inline(always)]
    fn mul(self, rhs: Vec3d<Cartesian>) -> Self::Output {
        (&self).mul(rhs)
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
        Self: Sized,
    {
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
        Self: Sized,
    {
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
        Self: Sized,
    {
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
