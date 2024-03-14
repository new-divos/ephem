use std::{
    convert::From,
    f64::consts::{FRAC_PI_2, PI},
    fmt,
    iter::{FromIterator, IntoIterator, Iterator},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::core::{consts::PI2, Canonizable, CrossMul, DotMul, FloatExt, Normalizable};

/// Trait representing a coordinate system.
///
/// This trait defines the basic properties and behavior that any coordinate system should have.
/// Types implementing this trait are expected to provide functionalities related to spatial
/// coordinates, transformations, and other operations specific to the coordinate system they
/// represent.
pub trait CoordinateSystem {
    /// Constant representing the first coordinate index.
    const E1_IDX: usize;

    /// Constant representing the second coordinate index.
    const E2_IDX: usize;

    /// Constant representing the third coordinate index.
    const E3_IDX: usize;
}

/// A trait for converting objects to Cartesian coordinates.
///
/// This trait defines a method `to_c` which converts an object to Cartesian coordinates represented by a `Vec3d`
/// in the Cartesian coordinate system.
pub trait ToCartesian {
    /// Converts the object to Cartesian coordinates.
    ///
    /// # Returns
    ///
    /// A `Vec3d` representing the Cartesian coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, ToCartesian, Vec3d};
    ///
    /// struct MyCoordinate {}
    ///
    /// impl ToCartesian for MyCoordinate {
    ///     fn to_c(&self) -> Vec3d<Cartesian> {
    ///         CartesianBuilder::new().build() // returning a placeholder value
    ///     }
    /// }
    ///
    /// let coordinate = MyCoordinate {};
    /// let cartesian_vector = coordinate.to_c();
    /// assert_eq!(cartesian_vector.x(), 0.0);
    /// assert_eq!(cartesian_vector.y(), 0.0);
    /// assert_eq!(cartesian_vector.z(), 0.0);
    /// ```
    fn to_c(&self) -> Vec3d<Cartesian>;
}

/// A trait for converting objects to cylindrical coordinates.
///
/// This trait defines a method `to_y` which converts an object to cylindrical coordinates represented by a `Vec3d`
/// in the cylindrical coordinate system.
pub trait ToCylindrical {
    /// Converts the object to cylindrical coordinates.
    ///
    /// # Returns
    ///
    /// A `Vec3d` representing the cylindrical coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, ToCylindrical, Vec3d};
    ///
    /// struct MyCoordinate {}
    ///
    /// impl ToCylindrical for MyCoordinate {
    ///     fn to_y(&self) -> Vec3d<Cylindrical> {
    ///         CylindricalBuilder::new().build() // returning a placeholder value
    ///     }
    /// }
    ///
    /// let coordinate = MyCoordinate {};
    /// let cylindrical_vector = coordinate.to_y();
    /// assert_eq!(cylindrical_vector.radius(), 0.0);
    /// assert_eq!(cylindrical_vector.azimuth(), 0.0);
    /// assert_eq!(cylindrical_vector.altitude(), 0.0);
    /// ```
    fn to_y(&self) -> Vec3d<Cylindrical>;
}

/// A trait for converting objects to spherical coordinates.
///
/// This trait defines a method `to_s` which converts an object to spherical coordinates represented by a `Vec3d`
/// in the spherical coordinate system.
pub trait ToSpherical {
    /// Converts the object to spherical coordinates.
    ///
    /// # Returns
    ///
    /// A `Vec3d` representing the spherical coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, ToSpherical, Vec3d};
    ///
    /// struct MyCoordinate {}
    ///
    /// impl ToSpherical for MyCoordinate {
    ///     fn to_s(&self) -> Vec3d<Spherical> {
    ///         SphericalBuilder::new().build() // returning a placeholder value
    ///     }
    /// }
    ///
    /// let coordinate = MyCoordinate {};
    /// let spherical_vector = coordinate.to_s();
    /// assert_eq!(spherical_vector.radius(), 0.0);
    /// assert_eq!(spherical_vector.azimuth(), 0.0);
    /// assert_eq!(spherical_vector.latitude(), 0.0);
    /// ```
    fn to_s(&self) -> Vec3d<Spherical>;
}

/// Three-dimensional vector struct parameterized by a coordinate system.
///
/// The `Vec3d` struct represents a three-dimensional vector with coordinates stored as an array of
/// three `f64` values. The choice of coordinate system is determined by the type parameter `S`, which
/// must implement the `CoordinateSystem` trait.
pub struct Vec3d<S: CoordinateSystem>(
    /// Array representing the three coordinates of the vector.
    pub(super) [f64; 3],
    /// PhantomData marker to tie the coordinate system type to the vector.
    pub(super) PhantomData<S>,
);

/// Implements methods to iterate over the components of a 3D vector (`Vec3d<S>`) in any coordinate system `S`.
///
/// This implementation provides methods to create both immutable and mutable iterators over the components
/// of the vector. The `iter()` method returns an immutable iterator (`Vec3dIter`) allowing read-only access
/// to the components, while the `iter_mut()` method returns a mutable iterator (`Vec3dMutIter`) allowing
/// modification of the components.
impl<S: CoordinateSystem> Vec3d<S> {
    /// Returns an immutable iterator over the components of the vector.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    ///
    /// for (idx, component) in vector.iter().enumerate() {
    ///     match idx {
    ///         0 => assert_eq!(vector.x(), component),
    ///         1 => assert_eq!(vector.y(), component),
    ///         2 => assert_eq!(vector.z(), component),
    ///         _ => continue,
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Vec3dIter<'_> {
        Vec3dIter {
            data: &self.0,
            cursor: 0,
        }
    }

    /// Returns a mutable iterator over the components of the vector.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    ///
    /// for component in vector.iter_mut() {
    ///     *component += 1.0;
    /// }
    ///
    /// assert_eq!(vector.x(), 2.0);
    /// assert_eq!(vector.y(), 3.0);
    /// assert_eq!(vector.z(), 4.0);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> Vec3dMutIter<'_> {
        Vec3dMutIter {
            data: &mut self.0,
            cursor: 0,
        }
    }
}

/// Implementation block for `Vec3d` for division by a scalar with checking for zero divisor.
impl<S> Vec3d<S>
where
    S: CoordinateSystem,
    Vec3d<S>: Div<f64, Output = Self>,
{
    /// Divides the `Vec3d` by a scalar value, checking for zero divisor.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to divide the `Vec3d` by.
    ///
    /// # Returns
    ///
    /// * `Some(Self)` - Result of the division if the divisor is not zero.
    /// * `None` - If the divisor is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let result = vector.checked_div(2.0);
    /// assert!(result.is_some());
    /// if let Some(result) = result {
    ///     assert_eq!(result.x(), 0.5);
    ///     assert_eq!(result.y(), 1.0);
    ///     assert_eq!(result.z(), 1.5);
    /// }
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let zero_divisor_result = vector.checked_div(0.0);
    /// assert!(zero_divisor_result.is_none());
    /// ```
    #[inline]
    pub fn checked_div(self, rhs: f64) -> Option<Self> {
        if rhs != 0.0 {
            Some(self / rhs)
        } else {
            None
        }
    }
}

/// Implements the conversion from a 3D vector (`Vec3d<S>`) in any coordinate system `S` to a slice of `f64`.
///
/// This implementation allows the 3D vector to be treated as a slice of `f64`, providing read-only access to its components.
impl<S: CoordinateSystem> AsRef<[f64]> for Vec3d<S> {
    /// Returns a slice of `f64` representing the components of the vector.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    /// let slice: &[f64] = vector.as_ref();
    /// assert_eq!(slice, &[1.0, 2.0, 3.0]);
    #[inline]
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

/// Implements the conversion from a mutable reference to a 3D vector (`&mut Vec3d<S>`) in any coordinate system `S`
/// to a mutable slice of `f64`.
///
/// This implementation allows the mutable reference to the 3D vector to be treated as a mutable slice of `f64`,
/// providing mutable access to its components.
impl<S: CoordinateSystem> AsMut<[f64]> for Vec3d<S> {
    /// Returns a mutable slice of `f64` representing the components of the vector.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    /// let slice: &mut [f64] = vector.as_mut();
    /// slice[0] = 4.0;
    /// assert_eq!(vector.x(), 4.0);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

/// Implementation block for cloning `Vec3d` vectors.
impl<S: CoordinateSystem> Clone for Vec3d<S> {
    /// Creates a deep copy of the vector.
    ///
    /// This method creates a new `Vec3d` with the same components as the original vector.
    ///
    /// # Returns
    ///
    /// A new `Vec3d` with the same components as the original vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let cloned_v = v.clone();
    /// assert_eq!(cloned_v.x(), 1.0);
    /// assert_eq!(cloned_v.y(), 2.0);
    /// assert_eq!(cloned_v.z(), 3.0);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        Self([self.0[0], self.0[1], self.0[2]], PhantomData::<S> {})
    }
}

/// Implementation block for comparing equality of `Vec3d` vectors.
impl<S: CoordinateSystem> PartialEq for Vec3d<S> {
    /// Compares equality of two `Vec3d` vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of both vectors are equal, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// assert_eq!(v1, v2);
    /// let v3 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// assert_ne!(v1, v3);
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0[0].eq(&other.0[0]) && self.0[1].eq(&other.0[1]) && self.0[2].eq(&other.0[2])
    }
}

/// Implementation of converting a 3-dimensional vector to a tuple of coordinates.
///
/// This implementation converts a 3-dimensional vector (`Vec3d`) representing coordinates
/// in a specified coordinate system (`CoordinateSystem`) to a tuple of `(f64, f64, f64)`.
/// The elements of the tuple correspond to the coordinates in the vector according to the
/// indices defined in the coordinate system.
impl<S: CoordinateSystem> From<Vec3d<S>> for (f64, f64, f64) {
    #[inline]
    /// Converts the given 3-dimensional vector to a tuple of coordinates.
    fn from(vector: Vec3d<S>) -> Self {
        (
            vector.0[S::E1_IDX],
            vector.0[S::E2_IDX],
            vector.0[S::E3_IDX],
        )
    }
}

/// Implementation block for multiplication of a scalar by a `Vec3d`.
impl<S> Mul<Vec3d<S>> for f64
where
    S: CoordinateSystem,
    Vec3d<S>: Mul<f64, Output = Vec3d<S>>,
{
    /// The type of the output when multiplying a scalar by a `Vec3d`.
    type Output = Vec3d<S>;

    /// Multiplies a scalar by a `Vec3d`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `Vec3d` to be multiplied by the scalar.
    ///
    /// # Returns
    ///
    /// The result of the multiplication operation, which is a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let scalar = 2.0;
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let result = scalar * vector;
    /// assert_eq!(result.x(), 2.0);
    /// assert_eq!(result.y(), 4.0);
    /// assert_eq!(result.z(), 6.0);
    /// ```
    #[inline]
    fn mul(self, rhs: Vec3d<S>) -> Self::Output {
        rhs.mul(self)
    }
}

/// Implements conversion from a 3D vector (`Vec3d<S>`) in any coordinate system `S`
/// to an iterator over its components (`Vec3dIntoIter`), consuming the vector in the process.
impl<S: CoordinateSystem> IntoIterator for Vec3d<S> {
    type Item = f64;
    type IntoIter = Vec3dIntoIter;

    /// Converts the 3D vector into an iterator over its components, consuming the vector in the process.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    /// let mut iter = vector.into_iter();
    /// assert_eq!(iter.next(), Some(1.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Vec3dIntoIter {
            data: self.0,
            cursor: 0,
        }
    }
}

/// Implements conversion from a reference to a 3D vector (`&Vec3d<S>`) in any coordinate system `S`
/// to an iterator over its components (`Vec3dIter`).
impl<'a, S: CoordinateSystem> IntoIterator for &'a Vec3d<S> {
    type Item = f64;
    type IntoIter = Vec3dIter<'a>;

    /// Converts the reference to a 3D vector into an iterator over its components.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    /// let mut t: Vec<f64> = Vec::new();
    /// for component in &vector {
    ///     t.push(component);
    /// }
    ///
    /// let u = CartesianBuilder::from_iter(t).build();
    /// assert_eq!(u, vector);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Implements conversion from a mutable reference to a 3D vector (`&mut Vec3d<S>`) in any coordinate system `S`
/// to an iterator over its mutable components (`Vec3dMutIter`).
impl<'a, S: CoordinateSystem> IntoIterator for &'a mut Vec3d<S> {
    type Item = &'a mut f64;
    type IntoIter = Vec3dMutIter<'a>;

    /// Converts the mutable reference to a 3D vector into an iterator over its mutable components.
    ///
    /// # Example
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut vector = CartesianBuilder::with(1.0, 2.0, 3.0).build(); // (x, y, z)
    /// for component in &mut vector {
    ///     *component += 1.0;
    /// }
    /// assert_eq!(vector, CartesianBuilder::with(2.0, 3.0, 4.0).build());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Iterator over the components of a 3D vector (`Vec3d<S>`).
///
/// This iterator allows iterating over the components of a 3D vector in a forward direction.
pub struct Vec3dIter<'a> {
    /// Reference to the underlying data containing the components of the vector.
    data: &'a [f64],
    /// Cursor indicating the current position within the data.
    cursor: usize,
}

impl<'a> Iterator for Vec3dIter<'a> {
    type Item = f64;

    /// Advances the iterator and returns the next component of the vector.
    /// Returns `None` when all components have been iterated.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if (0..self.data.len()).contains(&self.cursor) {
            let value = self.data[self.cursor];
            self.cursor += 1;

            Some(value)
        } else {
            None
        }
    }

    /// Returns the size hint of the iterator.
    ///
    /// This returns a tuple where the first element is the exact size of the iterator,
    /// and the second element is `Some(size)` representing the upper bound (same as the exact size).
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }

    /// Consumes the iterator and returns the last component of the vector.
    #[inline]
    fn last(self) -> Option<Self::Item> {
        Some(self.data[self.data.len() - 1])
    }
}

/// Mutable iterator over the components of a mutable reference to a 3D vector (`&mut Vec3d<S>`).
///
/// This iterator allows iterating over the mutable components of a 3D vector in a forward direction.
pub struct Vec3dMutIter<'a> {
    /// Mutable reference to the underlying data containing the components of the vector.
    data: &'a mut [f64],
    /// Cursor indicating the current position within the data.
    cursor: usize,
}

impl<'a> Iterator for Vec3dMutIter<'a> {
    type Item = &'a mut f64;

    /// Advances the iterator and returns a mutable reference to the next component of the vector.
    /// Returns `None` when all components have been iterated.
    fn next(&mut self) -> Option<Self::Item> {
        if (0..self.data.len()).contains(&self.cursor) {
            let i = self.cursor;
            self.cursor += 1;

            let data_ptr = self.data.as_mut_ptr();
            unsafe { Some(&mut *data_ptr.add(i)) }
        } else {
            None
        }
    }

    /// Returns the size hint of the iterator.
    ///
    /// This returns a tuple where the first element is the exact size of the iterator,
    /// and the second element is `Some(size)` representing the upper bound (same as the exact size).
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }
}

/// Iterator over the components of a 3D vector (`Vec3d`), consuming the vector in the process.
///
/// This iterator allows iterating over the components of a 3D vector in a forward direction, consuming the vector in the process.
pub struct Vec3dIntoIter {
    /// Array containing the components of the vector.
    data: [f64; 3],
    /// Cursor indicating the current position within the data.
    cursor: usize,
}

impl Iterator for Vec3dIntoIter {
    type Item = f64;

    /// Advances the iterator and returns the next component of the vector.
    /// Returns `None` when all components have been iterated.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if (0..=2).contains(&self.cursor) {
            let value = self.data[self.cursor];
            self.cursor += 1;

            Some(value)
        } else {
            None
        }
    }

    /// Returns the size hint of the iterator.
    ///
    /// This returns a tuple where the first element is the exact size of the iterator,
    /// and the second element is `Some(size)` representing the upper bound (same as the exact size).
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (3, Some(3))
    }

    /// Consumes the iterator and returns the last component of the vector.
    #[inline]
    fn last(self) -> Option<Self::Item> {
        Some(self.data[2])
    }
}

/// Struct representing the Cartesian coordinate system.
#[derive(Debug)]
pub struct Cartesian;

/// It also includes constants defining the indices for commonly used Cartesian
/// coordinates (X, Y, and Z).
impl Cartesian {
    /// Constant representing the X-coordinate index in Cartesian coordinates.
    const X_IDX: usize = 0;

    /// Constant representing the Y-coordinate index in Cartesian coordinates.
    const Y_IDX: usize = 1;

    /// Constant representing the Z-coordinate index in Cartesian coordinates.
    const Z_IDX: usize = 2;
}

/// The `Cartesian` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait.
impl CoordinateSystem for Cartesian {
    /// Constant representing the first coordinate index in Cartesian coordinates.
    const E1_IDX: usize = Self::X_IDX;

    /// Constant representing the second coordinate index in Cartesian coordinates.
    const E2_IDX: usize = Self::Y_IDX;

    /// Constant representing the third coordinate index in Cartesian coordinates.
    const E3_IDX: usize = Self::Z_IDX;
}

/// Additional methods for three-dimensional vectors in the Cartesian coordinate system.
///
/// This implementation block adds convenience methods to the `Vec3d` struct when the chosen
/// coordinate system is Cartesian. It provides direct access to the X, Y, and Z components of the
/// vector, as well as implements the `Vec3dNorm` trait to calculate the vector's magnitude.
impl Vec3d<Cartesian> {
    /// Returns the X-component of the three-dimensional vector.
    #[inline]
    pub fn x(&self) -> f64 {
        self.0[Cartesian::X_IDX]
    }

    /// Returns the Y-component of the three-dimensional vector.
    #[inline]
    pub fn y(&self) -> f64 {
        self.0[Cartesian::Y_IDX]
    }

    /// Returns the Z-component of the three-dimensional vector.
    #[inline]
    pub fn z(&self) -> f64 {
        self.0[Cartesian::Z_IDX]
    }
}

/// Implementation of the `Normalizable` trait for three-dimensional vectors in the Cartesian
/// coordinate system.
///
/// This allows the calculation of the magnitude (norm) of a `Vec3d<Cartesian>` vector.
impl Normalizable for Vec3d<Cartesian> {
    /// The type of the result of the norm calculation.
    type Output = f64;

    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> Self::Output {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]).sqrt()
    }
}

/// Implementation block for converting a Cartesian `Vec3d` vector to a cylindrical `Vec3d` vector.
///
/// This implementation block provides the implementation of the `to_y` method for converting a Cartesian `Vec3d` vector
/// to a cylindrical `Vec3d` vector. It calculates the radius and azimuth components of the cylindrical vector
/// based on the x and y components of the Cartesian vector.
impl ToCylindrical for Vec3d<Cartesian> {
    /// Converts a Cartesian `Vec3d` vector to a cylindrical `Vec3d` vector.
    ///
    /// # Returns
    ///
    /// A cylindrical `Vec3d` vector representing the same position as the original Cartesian vector.
    fn to_y(&self) -> Vec3d<Cylindrical> {
        let x = self.0[Cartesian::X_IDX];
        let y = self.0[Cartesian::Y_IDX];

        let phi = if x == 0.0 && y == 0.0 {
            0.0
        } else {
            y.atan2(x)
        };

        Vec3d::<Cylindrical>(
            [x.hypot(y), phi.fmod(PI2), self.0[Cartesian::Z_IDX]],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implementation block for converting a Cartesian `Vec3d` vector to a spherical `Vec3d` vector.
///
/// This implementation block provides the implementation of the `to_s` method for converting a Cartesian `Vec3d` vector
/// to a spherical `Vec3d` vector. It calculates the radius, azimuth, and latitude components of the spherical vector
/// based on the x, y, and z components of the Cartesian vector.
impl ToSpherical for Vec3d<Cartesian> {
    /// Converts a Cartesian `Vec3d` vector to a spherical `Vec3d` vector.
    ///
    /// # Returns
    ///
    /// A spherical `Vec3d` vector representing the same position as the original Cartesian vector.
    fn to_s(&self) -> Vec3d<Spherical> {
        let x = self.0[Cartesian::X_IDX];
        let y = self.0[Cartesian::Y_IDX];
        let z = self.0[Cartesian::Z_IDX];

        let rho_sq = x * x + y * y;
        let r = (rho_sq + z * z).sqrt();

        let phi = if x == 0.0 && y == 0.0 {
            0.0
        } else {
            y.atan2(x)
        };

        let rho = rho_sq.sqrt();
        let theta = if rho == 0.0 && z == 0.0 {
            0.0
        } else {
            z.atan2(rho)
        };

        Vec3d::<Spherical>([r, phi.fmod(PI2), theta], PhantomData::<Spherical> {})
    }
}

/// Implementation block for converting a `Vec3d` vector to a Cartesian `Vec3d` vector.
///
/// This implementation block provides the implementation of the `From` trait for converting a `Vec3d` vector
/// to a Cartesian `Vec3d` vector. It requires that the input vector type `S` implements the `CoordinateSystem` trait
/// and the `Vec3d<S>` type implements the `ToCartesian` trait, which provides a method to convert the vector
/// to Cartesian coordinates.
impl<S> From<Vec3d<S>> for Vec3d<Cartesian>
where
    S: CoordinateSystem,
    Vec3d<S>: ToCartesian,
{
    /// Converts a `Vec3d` vector to a Cartesian `Vec3d` vector.
    ///
    /// # Arguments
    ///
    /// * `vector` - The input vector of any coordinate system.
    ///
    /// # Returns
    ///
    /// A Cartesian `Vec3d` vector representing the same position as the input vector.
    #[inline]
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_c()
    }
}

/// Implementation block for negating a Cartesian vector.
impl Neg for Vec3d<Cartesian> {
    /// The type of the result of the negation operation.
    type Output = Self;

    /// Negates a Cartesian `Vec3d` vector.
    ///
    /// # Returns
    ///
    /// A new Cartesian `Vec3d` vector with negated components.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let negated_vector = -vector;
    /// assert_eq!(negated_vector.x(), -1.0);
    /// assert_eq!(negated_vector.y(), -2.0);
    /// assert_eq!(negated_vector.z(), -3.0);
    /// ```
    #[inline]
    fn neg(self) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                -self.0[Cartesian::X_IDX],
                -self.0[Cartesian::Y_IDX],
                -self.0[Cartesian::Z_IDX],
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implementation block for adding two Cartesian `Vec3d` vectors.
impl Add for Vec3d<Cartesian> {
    /// The type of the result of the addition operation.
    type Output = Self;

    /// Adds two Cartesian `Vec3d` vectors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector to be added to the left-hand side vector.
    ///
    /// # Returns
    ///
    /// A new Cartesian `Vec3d` vector with components equal to the sum of the corresponding
    /// components of the input vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// let sum = v1 + v2;
    /// assert_eq!(sum.x(), 5.0);
    /// assert_eq!(sum.y(), 7.0);
    /// assert_eq!(sum.z(), 9.0);
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::X_IDX] + rhs.0[Cartesian::X_IDX],
                self.0[Cartesian::Y_IDX] + rhs.0[Cartesian::Y_IDX],
                self.0[Cartesian::Z_IDX] + rhs.0[Cartesian::Z_IDX],
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implementation block for in-place addition of Cartesian `Vec3d` vectors.
impl AddAssign for Vec3d<Cartesian> {
    /// Performs in-place addition of Cartesian `Vec3d` vectors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector to be added to the left-hand side vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// v1 += v2;
    /// assert_eq!(v1.x(), 5.0);
    /// assert_eq!(v1.y(), 7.0);
    /// assert_eq!(v1.z(), 9.0);
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[Cartesian::X_IDX] += rhs.0[Cartesian::X_IDX];
        self.0[Cartesian::Y_IDX] += rhs.0[Cartesian::Y_IDX];
        self.0[Cartesian::Z_IDX] += rhs.0[Cartesian::Z_IDX];
    }
}

/// Implementation block for subtracting Cartesian `Vec3d` vectors.
impl Sub for Vec3d<Cartesian> {
    /// The type of the result of the substraction operation.
    type Output = Self;

    /// Subtracts two Cartesian `Vec3d` vectors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector to be subtracted from the left-hand side vector.
    ///
    /// # Returns
    ///
    /// A new Cartesian `Vec3d` vector with components equal to the difference between the corresponding components
    /// of the input vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// let difference = v1 - v2;
    /// assert_eq!(difference.x(), -3.0);
    /// assert_eq!(difference.y(), -3.0);
    /// assert_eq!(difference.z(), -3.0);
    /// ```

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::X_IDX] - rhs.0[Cartesian::X_IDX],
                self.0[Cartesian::Y_IDX] - rhs.0[Cartesian::Y_IDX],
                self.0[Cartesian::Z_IDX] - rhs.0[Cartesian::Z_IDX],
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implementation block for in-place subtraction of Cartesian `Vec3d` vectors.
impl SubAssign for Vec3d<Cartesian> {
    /// Subtracts the components of another Cartesian `Vec3d` vector from the components of this vector.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector whose components are subtracted from the components of the left-hand side vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    /// v1 -= v2;
    /// assert_eq!(v1.x(), -3.0);
    /// assert_eq!(v1.y(), -3.0);
    /// assert_eq!(v1.z(), -3.0);
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[Cartesian::X_IDX] -= rhs.0[Cartesian::X_IDX];
        self.0[Cartesian::Y_IDX] -= rhs.0[Cartesian::Y_IDX];
        self.0[Cartesian::Z_IDX] -= rhs.0[Cartesian::Z_IDX];
    }
}

/// Implementation block for scalar multiplication of Cartesian `Vec3d` vectors.
impl Mul<f64> for Vec3d<Cartesian> {
    /// The type of the result of the multiplication by a floating-point scalar operation.
    type Output = Self;

    /// Multiplies each component of the vector by a floating-point scalar value.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The floating-point scalar value to multiply with each component of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let result = v * 2.0;
    /// assert_eq!(result.x(), 2.0);
    /// assert_eq!(result.y(), 4.0);
    /// assert_eq!(result.z(), 6.0);
    /// ```
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::X_IDX] * rhs,
                self.0[Cartesian::Y_IDX] * rhs,
                self.0[Cartesian::Z_IDX] * rhs,
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implementation block for multiplying `Vec3d<Cartesian>` vectors by a scalar in-place.
impl MulAssign<f64> for Vec3d<Cartesian> {
    /// Multiplies the vector components by the scalar in-place.
    ///
    /// This method modifies the components of the vector by multiplying them with the given scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to multiply the vector components by.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut v = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// v *= 2.0;
    /// assert_eq!(v.x(), 2.0);
    /// assert_eq!(v.y(), 4.0);
    /// assert_eq!(v.z(), 6.0);
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0[Cartesian::X_IDX] *= rhs;
        self.0[Cartesian::Y_IDX] *= rhs;
        self.0[Cartesian::Z_IDX] *= rhs;
    }
}

/// Implementation block for scalar division of Cartesian `Vec3d` vectors.
impl Div<f64> for Vec3d<Cartesian> {
    /// The type of the result of the division by a floating-point scalar operation.
    type Output = Self;

    /// Divides each component of the vector by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to divide each component by.
    ///
    /// # Returns
    ///
    /// The resulting vector after dividing each component by the scalar value.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let result = v / 2.0;
    /// assert_eq!(result.x(), 0.5);
    /// assert_eq!(result.y(), 1.0);
    /// assert_eq!(result.z(), 1.5);
    /// ```
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::X_IDX] / rhs,
                self.0[Cartesian::Y_IDX] / rhs,
                self.0[Cartesian::Z_IDX] / rhs,
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implementation block for dividing `Vec3d<Cartesian>` vectors by a scalar in-place.
impl DivAssign<f64> for Vec3d<Cartesian> {
    /// Divides the vector components by the scalar in-place.
    ///
    /// This method modifies the components of the vector by dividing them with the given scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to divide the vector components by.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let mut v = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// v /= 2.0;
    /// assert_eq!(v.x(), 0.5);
    /// assert_eq!(v.y(), 1.0);
    /// assert_eq!(v.z(), 1.5);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.0[Cartesian::X_IDX] /= rhs;
        self.0[Cartesian::Y_IDX] /= rhs;
        self.0[Cartesian::Z_IDX] /= rhs;
    }
}

/// Implements dot product multiplication for Cartesian vectors.
impl DotMul for Vec3d<Cartesian> {
    /// The output type of the dot product operation.
    type Output = f64;

    /// Computes the dot product of two Cartesian vectors.
    ///
    /// # Arguments
    ///
    /// * `self` - The first Cartesian vector.
    /// * `rhs` - The second Cartesian vector.
    ///
    /// # Returns
    ///
    /// The dot product of the two Cartesian vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::DotMul;
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v1 = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let v2 = CartesianBuilder::with(4.0, 5.0, 6.0).build();
    ///
    /// let dot_product = v1.dot(v2);
    /// assert_eq!(dot_product, 32.0);
    /// ```
    #[inline]
    fn dot(self, rhs: Self) -> Self::Output {
        self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1] + self.0[2] * rhs.0[2]
    }
}

/// Implements cross product multiplication for Cartesian vectors.
impl CrossMul for Vec3d<Cartesian> {
    /// The output type of the cross product operation.
    type Output = Self;

    /// Computes the cross product of two Cartesian vectors.
    ///
    /// # Arguments
    ///
    /// * `self` - The first Cartesian vector.
    /// * `rhs` - The second Cartesian vector.
    ///
    /// # Returns
    ///
    /// The cross product of the two Cartesian vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::CrossMul;
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let v1 = CartesianBuilder::with(-2.0, 3.0, 0.0).build();
    /// let v2 = CartesianBuilder::with(-2.0, 0.0, 6.0).build();
    ///
    /// let cross_product = v1.cross(v2);
    /// let v3 = CartesianBuilder::with(18.0, 12.0, 6.0).build();
    /// assert_eq!(cross_product, v3);
    /// ```
    #[inline]
    fn cross(self, rhs: Self) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::Y_IDX] * rhs.0[Cartesian::Z_IDX]
                    - self.0[Cartesian::Z_IDX] * rhs.0[Cartesian::Y_IDX],
                self.0[Cartesian::Z_IDX] * rhs.0[Cartesian::X_IDX]
                    - self.0[Cartesian::X_IDX] * rhs.0[Cartesian::Z_IDX],
                self.0[Cartesian::X_IDX] * rhs.0[Cartesian::Y_IDX]
                    - self.0[Cartesian::Y_IDX] * rhs.0[Cartesian::X_IDX],
            ],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implements the `Debug` trait for Cartesian vectors.
impl fmt::Debug for Vec3d<Cartesian> {
    /// Formats the Cartesian vector using the `Debug` trait.
    ///
    /// This function formats the Cartesian vector as a debug string containing the x, y, and z components.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to a formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating success or failure in formatting the Cartesian vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let vector = CartesianBuilder::with(1.0, 2.0, 3.0).build();
    /// let s = format!("{:?}", vector);
    /// assert_eq!(s.as_str(), "Vec3d { x: 1.0, y: 2.0, z: 3.0 }");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vec3d")
            .field("x", &self.0[Cartesian::X_IDX])
            .field("y", &self.0[Cartesian::Y_IDX])
            .field("z", &self.0[Cartesian::Z_IDX])
            .finish()
    }
}

/// Builder struct for creating instances of the vectors in Cartesian coordinate system.
///
/// The `CartesianBuilder` struct facilitates the construction of Vec3d instances
/// with various methods to set specific coordinates or generate unit vectors along the axes.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianBuilder {
    /// X component of the cartesian vector.
    x: f64,

    /// Y component of the cartesian vector.
    y: f64,

    /// Z component of the cartesian vector.
    z: f64,
}

impl CartesianBuilder {
    /// Creates a new builder instance with default values (0.0 for each coordinate).
    #[inline]
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a builder instance with specified cartesian coordinates.
    #[inline]
    pub fn with(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Creates a builder instance representing the unit vector along the X-axis.
    #[inline]
    pub fn unit_x() -> Self {
        Self {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a builder instance representing the unit vector along the Y-axis.
    #[inline]
    pub fn unit_y() -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    /// Creates a builder instance representing the unit vector along the Z-axis.
    #[inline]
    pub fn unit_z() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    /// Sets the X-coordinate and returns a builder for chaining.
    #[inline]
    pub fn x(mut self, value: f64) -> Self {
        self.x = value;
        self
    }

    /// Sets the Y-coordinate and returns a builder for chaining.
    #[inline]
    pub fn y(mut self, value: f64) -> Self {
        self.y = value;
        self
    }

    /// Sets the Z-coordinate and returns a builder for chaining.
    #[inline]
    pub fn z(mut self, value: f64) -> Self {
        self.z = value;
        self
    }

    /// Builds and returns a `Vec3d<Cartesian>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Cartesian> {
        Vec3d::<Cartesian>([self.x, self.y, self.z], PhantomData::<Cartesian> {})
    }
}

impl Default for CartesianBuilder {
    /// Creates a default `CartesianBuilder` instance with default values (0.0 for each coordinate).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Implements conversion from an iterator over `f64` values into a Cartesian vector builder.
impl FromIterator<f64> for CartesianBuilder {
    /// Constructs a Cartesian vector builder from an iterator over `f64` values.
    ///
    /// This function consumes the iterator and constructs a Cartesian vector builder from the first three `f64` values
    /// encountered in the iterator. If the iterator contains fewer than three values, the remaining components are set to zero.
    /// If the iterator contains more than three values, only the first three values are used to construct the vector.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator over `f64` values.
    ///
    /// # Returns
    ///
    /// A Cartesian vector builder constructed from the first three `f64` values encountered in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::iter::FromIterator;
    /// use ephem::core::vectors::{Cartesian, CartesianBuilder, Vec3d};
    ///
    /// let iter = vec![1.0, 2.0, 3.0, 4.0].into_iter();
    /// let builder = CartesianBuilder::from_iter(iter);
    /// let vector = builder.build();
    ///
    /// assert_eq!(vector.x(), 1.0);
    /// assert_eq!(vector.y(), 2.0);
    /// assert_eq!(vector.z(), 3.0);
    /// ```
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (idx, value) in iter.into_iter().enumerate() {
            match idx {
                Cartesian::X_IDX => x = value,
                Cartesian::Y_IDX => y = value,
                Cartesian::Z_IDX => z = value,
                _ => continue,
            }
        }

        Self::with(x, y, z)
    }
}

/// Struct representing the Cylindrical coordinate system.
#[derive(Debug)]
pub struct Cylindrical;

/// It also includes constants defining the indices for commonly used Cylindrical
/// coordinates (Radius, Azimuth, and Altitude).
impl Cylindrical {
    /// Constant representing the Radius-coordinate index in Cylindrical coordinates.
    const RADIUS_IDX: usize = 0;

    /// Constant representing the Azimuth-coordinate index in Cylindrical coordinates.
    const AZIMUTH_IDX: usize = 1;

    /// Constant representing the Altitude-coordinate index in Cylindrical coordinates.
    const ALTITUDE_IDX: usize = 2;
}

/// The `Cylindrical` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait.
impl CoordinateSystem for Cylindrical {
    /// Constant representing the first coordinate index in Cylindrical coordinates.
    const E1_IDX: usize = Self::RADIUS_IDX;

    /// Constant representing the second coordinate index in Cylindrical coordinates.
    const E2_IDX: usize = Self::AZIMUTH_IDX;

    /// Constant representing the third coordinate index in Cylindrical coordinates.
    const E3_IDX: usize = Self::ALTITUDE_IDX;
}

/// Additional methods for three-dimensional vectors in the cylindrical coordinate system.
///
/// This implementation block adds convenience methods to the `Vec3d` struct when the chosen
/// coordinate system is cylindrical. It provides direct access to the radius, azimuth, and altitude
/// components of the vector.
impl Vec3d<Cylindrical> {
    /// Returns the radius component of the three-dimensional vector.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.0[Cylindrical::RADIUS_IDX]
    }

    /// Returns the azimuth component of the three-dimensional vector.
    #[inline]
    pub fn azimuth(&self) -> f64 {
        self.0[Cylindrical::AZIMUTH_IDX]
    }

    /// Returns the altitude component of the three-dimensional vector.
    #[inline]
    pub fn altitude(&self) -> f64 {
        self.0[Cylindrical::ALTITUDE_IDX]
    }
}

/// Implementation of the `Normalizable` trait for three-dimensional vectors in the cylindrical
/// coordinate system.
///
/// This allows the calculation of the norm (magnitude) of a `Vec3d<Cylindrical>` vector.
impl Normalizable for Vec3d<Cylindrical> {
    /// The type of the result of the norm calculation.
    type Output = f64;

    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> Self::Output {
        (self.0[Cylindrical::RADIUS_IDX] * self.0[Cylindrical::RADIUS_IDX]
            + self.0[Cylindrical::ALTITUDE_IDX] * self.0[Cylindrical::ALTITUDE_IDX])
            .sqrt()
    }
}

/// Implements a method to convert a vector in cylindrical coordinates to its canonical form.
impl Canonizable for Vec3d<Cylindrical> {
    /// Converts the vector to its canonical form in cylindrical coordinates.
    ///
    /// If the radius component of the vector is negative, it negates the radius and adjusts
    /// the azimuth by adding π radians to maintain the equivalent direction.
    ///
    /// # Returns
    ///
    /// The vector in its canonical form.
    fn canonic(self) -> Self {
        let mut radius = self.0[Cylindrical::RADIUS_IDX];
        let mut azimuth = self.0[Cylindrical::AZIMUTH_IDX];

        // Adjust radius and azimuth if radius is negative
        if radius < 0.0 {
            radius = -radius;
            azimuth += PI;
        }

        // Ensure azimuth is within the range [0, 2π)
        Vec3d::<Cylindrical>(
            [radius, azimuth.fmod(PI2), self.0[Cylindrical::ALTITUDE_IDX]],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implements a method to convert a vector in cylindrical coordinates to Cartesian coordinates.
impl ToCartesian for Vec3d<Cylindrical> {
    /// Converts the vector from cylindrical coordinates to Cartesian coordinates.
    ///
    /// # Returns
    ///
    /// The vector converted to Cartesian coordinates.
    fn to_c(&self) -> Vec3d<Cartesian> {
        let (sa, ca) = self.0[Cylindrical::AZIMUTH_IDX].sin_cos();
        let rho = self.0[Cylindrical::RADIUS_IDX];

        // Calculate Cartesian components from cylindrical components
        Vec3d::<Cartesian>(
            [ca * rho, sa * rho, self.0[Cylindrical::ALTITUDE_IDX]],
            PhantomData::<Cartesian> {},
        )
    }
}

/// Implements a method to convert a vector in cylindrical coordinates to spherical coordinates.
impl ToSpherical for Vec3d<Cylindrical> {
    /// Converts the vector from cylindrical coordinates to spherical coordinates.
    ///
    /// # Returns
    ///
    /// The vector converted to spherical coordinates.
    fn to_s(&self) -> Vec3d<Spherical> {
        let rho = self.0[Cylindrical::RADIUS_IDX];
        let z = self.0[Cylindrical::ALTITUDE_IDX];

        // Calculate spherical components from cylindrical components
        let r = (rho * rho + z * z).sqrt();
        let theta = if rho == 0.0 && z == 0.0 {
            0.0
        } else {
            z.atan2(rho)
        };

        Vec3d::<Spherical>(
            [r, self.0[Cylindrical::AZIMUTH_IDX], theta],
            PhantomData::<Spherical> {},
        )
    }
}

/// Implements a conversion from a vector in any coordinate system to cylindrical coordinates.
impl<S> From<Vec3d<S>> for Vec3d<Cylindrical>
where
    S: CoordinateSystem,
    Vec3d<S>: ToCylindrical,
{
    /// Converts a vector from any coordinate system to cylindrical coordinates.
    ///
    /// # Arguments
    ///
    /// * `vector` - The input vector to be converted.
    ///
    /// # Returns
    ///
    /// The vector converted to cylindrical coordinates.
    #[inline]
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_y()
    }
}

/// Implements the negation operation for vectors in cylindrical coordinates.
impl Neg for Vec3d<Cylindrical> {
    /// The type of the result of the negation operation.
    type Output = Self;

    /// Negates each component of the vector in cylindrical coordinates.
    ///
    /// # Arguments
    ///
    /// * `self` - The vector to be negated.
    ///
    /// # Returns
    ///
    /// A new vector with negated components.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let vector = CylindricalBuilder::with(1.0, PI, 3.0).build();
    /// let negated_vector = -vector;
    /// assert_eq!(negated_vector.radius(), -1.0);
    /// assert_eq!(negated_vector.azimuth(), PI);
    /// assert_eq!(negated_vector.altitude(), -3.0);
    /// ```
    #[inline]
    fn neg(self) -> Self::Output {
        Vec3d::<Cylindrical>(
            [
                -self.0[Cylindrical::RADIUS_IDX],
                self.0[Cylindrical::AZIMUTH_IDX],
                -self.0[Cylindrical::ALTITUDE_IDX],
            ],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implements the multiplication operation between a cylindrical vector and a scalar.
///
/// This implementation performs element-wise multiplication of the cylindrical vector's components
/// by the given scalar value.
impl Mul<f64> for Vec3d<Cylindrical> {
    /// The type of the result of the multiplication by a floating-point scalar operation.
    type Output = Self;

    /// Performs the multiplication operation.
    ///
    /// # Parameters
    ///
    /// - `self`: The cylindrical vector.
    /// - `rhs`: The scalar value to multiply by.
    ///
    /// # Returns
    ///
    /// The resulting cylindrical vector after element-wise multiplication with the scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let v = CylindricalBuilder::with(2.0, 0.0, 3.0).build();
    /// let scalar = 2.5;
    /// let result = v * scalar;
    /// assert_eq!(result.radius(), 5.0);
    /// assert_eq!(result.azimuth(), 0.0);
    /// assert_eq!(result.altitude(), 7.5);
    /// ```
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Vec3d::<Cylindrical>(
            [
                self.0[Cylindrical::RADIUS_IDX] * rhs,
                self.0[Cylindrical::AZIMUTH_IDX],
                self.0[Cylindrical::ALTITUDE_IDX] * rhs,
            ],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implements the multiplication assignment operation between a cylindrical vector and a scalar.
///
/// This implementation performs element-wise multiplication of the cylindrical vector's radius and altitude
/// components by the given scalar value.
impl MulAssign<f64> for Vec3d<Cylindrical> {
    /// Performs the multiplication assignment operation.
    ///
    /// # Parameters
    ///
    /// - `self`: A mutable reference to the cylindrical vector.
    /// - `rhs`: The scalar value to multiply by.
    ///
    /// # Example
    ///
    /// ```
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let mut v = CylindricalBuilder::with(2.0, 0.0, 3.0).build();
    /// let scalar = 2.5;
    /// v *= scalar;
    /// assert_eq!(v.radius(), 5.0);
    /// assert_eq!(v.azimuth(), 0.0);
    /// assert_eq!(v.altitude(), 7.5);
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0[Cylindrical::RADIUS_IDX] *= rhs;
        self.0[Cylindrical::ALTITUDE_IDX] *= rhs;
    }
}

/// Implements the division operation between a cylindrical vector and a scalar.
///
/// This implementation performs element-wise division of the cylindrical vector's radius and altitude
/// components by the given scalar value.
impl Div<f64> for Vec3d<Cylindrical> {
    /// The type of the result of the division by a floating-point scalar operation.
    type Output = Self;

    /// Performs the division operation.
    ///
    /// # Parameters
    ///
    /// - `self`: The cylindrical vector.
    /// - `rhs`: The scalar value to divide by.
    ///
    /// # Returns
    ///
    /// A new cylindrical vector where each component is divided by the scalar value.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let v = CylindricalBuilder::with(5.0, 0.0, 7.5).build();
    /// let scalar = 2.5;
    /// let result = v / scalar;
    /// assert_eq!(result.radius(), 2.0);
    /// assert_eq!(result.azimuth(), 0.0);
    /// assert_eq!(result.altitude(), 3.0);
    /// ```
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Vec3d::<Cylindrical>(
            [
                self.0[Cylindrical::RADIUS_IDX] / rhs,
                self.0[Cylindrical::AZIMUTH_IDX],
                self.0[Cylindrical::ALTITUDE_IDX] / rhs,
            ],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implements the division assignment operation between a cylindrical vector and a scalar.
///
/// This implementation performs element-wise division of the cylindrical vector's radius and altitude
/// components by the given scalar value, modifying the vector in place.
impl DivAssign<f64> for Vec3d<Cylindrical> {
    /// Performs the division assignment operation.
    ///
    /// # Parameters
    ///
    /// - `self`: A mutable reference to the cylindrical vector.
    /// - `rhs`: The scalar value to divide by.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let mut v = CylindricalBuilder::with(5.0, 0.0, 7.5).build();
    /// let scalar = 2.5;
    /// v /= scalar;
    /// assert_eq!(v.radius(), 2.0);
    /// assert_eq!(v.azimuth(), 0.0);
    /// assert_eq!(v.altitude(), 3.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.0[Cylindrical::RADIUS_IDX] /= rhs;
        self.0[Cylindrical::ALTITUDE_IDX] /= rhs;
    }
}

/// Implements the `Debug` trait for Cylindrical vectors.
impl fmt::Debug for Vec3d<Cylindrical> {
    /// Formats the Cylindrical vector using the `Debug` trait.
    ///
    /// This function formats the Cylindrical vector as a debug string containing the radius, azimuth, and altitude components.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to a formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating success or failure in formatting the Cylindrical vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let vector = CylindricalBuilder::with(1.0, 2.0, 3.0).build();
    /// let s = format!("{:?}", vector);
    /// assert_eq!(s.as_str(), "Vec3d { radius: 1.0, azimuth: 2.0, altitude: 3.0 }");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vec3d")
            .field("radius", &self.0[Cylindrical::RADIUS_IDX])
            .field("azimuth", &self.0[Cylindrical::AZIMUTH_IDX])
            .field("altitude", &self.0[Cylindrical::ALTITUDE_IDX])
            .finish()
    }
}

/// Builder struct for creating instances of vectors in the cylindrical coordinate system.
///
/// The `CylindricalBuilder` struct facilitates the construction of vectors in the cylindrical
/// coordinate system with various methods to set specific coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct CylindricalBuilder {
    /// Radius component of the cylindrical vector.
    radius: f64,

    /// Azimuth component of the cylindrical vector.
    azimuth: f64,

    /// Altitude component of the cylindrical vector.
    altitude: f64,
}

impl CylindricalBuilder {
    /// Creates a new `CylindricalBuilder` instance with default values (0.0 for each component).
    #[inline]
    pub fn new() -> Self {
        Self {
            radius: 0.0,
            azimuth: 0.0,
            altitude: 0.0,
        }
    }

    /// Creates a `CylindricalBuilder` instance with specified coordinates.
    #[inline]
    pub fn with(radius: f64, azimuth: f64, altitude: f64) -> Self {
        Self {
            radius,
            azimuth,
            altitude,
        }
    }

    /// Sets the radius component and returns a builder for chaining.
    #[inline]
    pub fn radius(mut self, value: f64) -> Self {
        self.radius = value;
        self
    }

    /// Sets the azimuth component and returns a builder for chaining.
    #[inline]
    pub fn azimuth(mut self, value: f64) -> Self {
        self.azimuth = value;
        self
    }

    /// Sets the altitude component and returns a builder for chaining.
    #[inline]
    pub fn altitude(mut self, value: f64) -> Self {
        self.altitude = value;
        self
    }

    /// Builds and returns a `Vec3d<Cylindrical>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Cylindrical> {
        Vec3d::<Cylindrical>(
            [self.radius, self.azimuth, self.altitude],
            PhantomData::<Cylindrical> {},
        )
    }
}

impl Default for CylindricalBuilder {
    /// Creates a default `CylindricalBuilder` instance with default values (0.0 for each component).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Implements conversion from an iterator over `f64` values into a Cylindrical vector builder.
impl FromIterator<f64> for CylindricalBuilder {
    /// Constructs a Cylindrical vector builder from an iterator over `f64` values.
    ///
    /// This function consumes the iterator and constructs a Cylindrical vector builder from the first three `f64` values
    /// encountered in the iterator. If the iterator contains fewer than three values, the remaining components are set to zero.
    /// If the iterator contains more than three values, only the first three values are used to construct the vector.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator over `f64` values.
    ///
    /// # Returns
    ///
    /// A Cylindrical vector builder constructed from the first three `f64` values encountered in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::iter::FromIterator;
    /// use ephem::core::vectors::{Cylindrical, CylindricalBuilder, Vec3d};
    ///
    /// let iter = vec![1.0, 2.0, 3.0, 4.0].into_iter();
    /// let builder = CylindricalBuilder::from_iter(iter);
    /// let vector = builder.build();
    ///
    /// assert_eq!(vector.radius(), 1.0);
    /// assert_eq!(vector.azimuth(), 2.0);
    /// assert_eq!(vector.altitude(), 3.0);
    /// ```
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut radius = 0.0;
        let mut azimuth = 0.0;
        let mut altitude = 0.0;

        for (idx, value) in iter.into_iter().enumerate() {
            match idx {
                Cylindrical::RADIUS_IDX => radius = value,
                Cylindrical::AZIMUTH_IDX => azimuth = value,
                Cylindrical::ALTITUDE_IDX => altitude = value,
                _ => continue,
            }
        }

        Self::with(radius, azimuth, altitude)
    }
}

/// Struct representing the Spherical coordinate system.
#[derive(Debug)]
pub struct Spherical;

/// It also includes constants defining the indices for commonly used Spherical
/// coordinates (Radius, Azimuth, and Latitude).
impl Spherical {
    /// Constant representing the Radius-coordinate index in Spherical coordinates.
    const RADIUS_IDX: usize = 0;

    /// Constant representing the Azimuth-coordinate index in Spherical coordinates.
    const AZIMUTH_IDX: usize = 1;

    /// Constant representing the Latitude-coordinate index in Spherical coordinates.
    const LATITUDE_IDX: usize = 2;
}

/// The `Spherical` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait.
impl CoordinateSystem for Spherical {
    /// Constant representing the first coordinate index in Spherical coordinates.
    const E1_IDX: usize = Self::RADIUS_IDX;

    /// Constant representing the second coordinate index in Spherical coordinates.
    const E2_IDX: usize = Self::AZIMUTH_IDX;

    /// Constant representing the third coordinate index in Spherical coordinates.
    const E3_IDX: usize = Self::LATITUDE_IDX;
}

/// Additional methods for three-dimensional vectors in the spherical coordinate system.
///
/// This implementation block adds convenience methods to the `Vec3d` struct when the chosen
/// coordinate system is spherical. It provides direct access to the radius, azimuth, and latitude
/// components of the vector.
impl Vec3d<Spherical> {
    /// Returns the radius component of the three-dimensional vector.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.0[Spherical::RADIUS_IDX]
    }

    /// Returns the azimuth component of the three-dimensional vector.
    #[inline]
    pub fn azimuth(&self) -> f64 {
        self.0[Spherical::AZIMUTH_IDX]
    }

    /// Returns the latitude component of the three-dimensional vector.
    #[inline]
    pub fn latitude(&self) -> f64 {
        self.0[Spherical::LATITUDE_IDX]
    }
}

/// Implementation of the `Normalizable` trait for three-dimensional vectors in the spherical
/// coordinate system.
///
/// This allows the calculation of the norm (magnitude) of a `Vec3d<Spherical>` vector.
impl Normalizable for Vec3d<Spherical> {
    /// The type of the result of the norm calculation.
    type Output = f64;

    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> Self::Output {
        self.0[Spherical::RADIUS_IDX].abs()
    }
}

/// Provides a method to transform a spherical vector into a canonical form.
impl Canonizable for Vec3d<Spherical> {
    /// Transforms the `Vec3d<Spherical>` into its canonical form.
    ///
    /// This implementation for `Vec3d<Spherical>` allows transforming a spherical vector into a canonical form,
    /// ensuring that the radius is non-negative and the latitude is within the range [-π/2, π/2].
    /// If the radius is negative, it negates the radius, adjusts the azimuth by adding π, and flips the latitude.
    /// It further ensures that the latitude is within the specified range, adjusting it as necessary.
    fn canonic(self) -> Self {
        let mut radius = self.0[Spherical::RADIUS_IDX];
        let mut azimuth = self.0[Spherical::AZIMUTH_IDX];
        let mut latitude = self.0[Spherical::LATITUDE_IDX];

        // Ensure radius is non-negative
        if radius < 0.0 {
            radius = -radius;
            azimuth += PI;
            latitude = -latitude;
        }

        // Ensure latitude is within [-π/2, π/2]
        if !(-FRAC_PI_2..=FRAC_PI_2).contains(&latitude) {
            latitude = latitude.fmod(PI2);
            if latitude > FRAC_PI_2 {
                latitude = PI - latitude;
                if latitude < -FRAC_PI_2 {
                    latitude = -(PI + latitude);
                }
            }
        }

        // Return the vector in its canonical form
        Self(
            [radius, azimuth.fmod(PI2), latitude],
            PhantomData::<Spherical> {},
        )
    }
}

/// Converts a spherical vector to a Cartesian vector.
impl ToCartesian for Vec3d<Spherical> {
    /// Converts the vector from spherical coordinates to Cartesian coordinates.
    ///
    /// # Returns
    ///
    /// The vector converted to Cartesian coordinates.
    fn to_c(&self) -> Vec3d<Cartesian> {
        let (sa, ca) = self.0[Spherical::AZIMUTH_IDX].sin_cos();
        let (sl, cl) = self.0[Spherical::LATITUDE_IDX].sin_cos();
        let r = self.0[Spherical::RADIUS_IDX];
        let rho = r * cl;

        Vec3d::<Cartesian>([ca * rho, sa * rho, r * sl], PhantomData::<Cartesian> {})
    }
}

/// Converts a spherical vector to a Cylindrical vector.
impl ToCylindrical for Vec3d<Spherical> {
    /// Converts the vector from spherical coordinates to Cylindrical coordinates.
    ///
    /// # Returns
    ///
    /// The vector converted to Cylindrical coordinates.
    fn to_y(&self) -> Vec3d<Cylindrical> {
        let (sl, cl) = self.0[Spherical::LATITUDE_IDX].sin_cos();
        let r = self.0[Spherical::RADIUS_IDX];

        Vec3d::<Cylindrical>(
            [cl * r, self.0[Spherical::AZIMUTH_IDX], sl * r],
            PhantomData::<Cylindrical> {},
        )
    }
}

/// Implements a conversion from a vector in any coordinate system to spherical coordinates.
impl<S> From<Vec3d<S>> for Vec3d<Spherical>
where
    S: CoordinateSystem,
    Vec3d<S>: ToSpherical,
{
    /// Converts a vector from any coordinate system to spherical coordinates.
    ///
    /// # Arguments
    ///
    /// * `vector` - The input vector to be converted.
    ///
    /// # Returns
    ///
    /// The vector converted to spherical coordinates.
    #[inline]
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_s()
    }
}

/// Implements unary negation for Spherical vectors.
impl Neg for Vec3d<Spherical> {
    /// The type of the result of the negation operation.
    type Output = Self;

    /// Negates the Spherical vector.
    ///
    /// This function returns a new Spherical vector where the radius component is negated, and the azimuth and latitude
    /// components remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `self` - The Spherical vector to be negated.
    ///
    /// # Returns
    ///
    /// A new Spherical vector with the radius component negated and other components unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Neg;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let vector = SphericalBuilder::with(1.0, 2.0, 3.0).build();
    /// let negated_vector = -vector;
    ///
    /// assert_eq!(negated_vector.radius(), -1.0);
    /// assert_eq!(negated_vector.azimuth(), 2.0);
    /// assert_eq!(negated_vector.latitude(), 3.0);
    /// ```
    #[inline]
    fn neg(self) -> Self::Output {
        Self(
            [
                -self.0[Spherical::RADIUS_IDX],
                self.0[Spherical::AZIMUTH_IDX],
                self.0[Spherical::LATITUDE_IDX],
            ],
            PhantomData::<Spherical> {},
        )
    }
}

/// Implements scalar multiplication for Spherical vectors.
impl Mul<f64> for Vec3d<Spherical> {
    /// The type of the result of the multiplication by a floating-point scalar operation.
    type Output = Self;

    /// Multiplies the Spherical vector by a scalar.
    ///
    /// This function returns a new Spherical vector where each component is multiplied by the given scalar,
    /// except for the azimuth and latitude components, which remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `self` - The Spherical vector to be multiplied.
    /// * `rhs` - The scalar value to multiply by.
    ///
    /// # Returns
    ///
    /// A new Spherical vector with each component multiplied by the scalar, except for the azimuth and latitude components
    /// which remain unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Mul;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let vector = SphericalBuilder::with(1.0, 2.0, 3.0).build();
    /// let multiplied_vector = vector * 2.0;
    ///
    /// assert_eq!(multiplied_vector.radius(), 2.0);
    /// assert_eq!(multiplied_vector.azimuth(), 2.0);
    /// assert_eq!(multiplied_vector.latitude(), 3.0);
    /// ```
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self(
            [
                self.0[Spherical::RADIUS_IDX] * rhs,
                self.0[Spherical::AZIMUTH_IDX],
                self.0[Spherical::LATITUDE_IDX],
            ],
            PhantomData::<Spherical> {},
        )
    }
}

/// Implements in-place scalar multiplication for Spherical vectors.
impl MulAssign<f64> for Vec3d<Spherical> {
    /// Multiplies the Spherical vector in-place by a scalar.
    ///
    /// This function multiplies the radius component of the Spherical vector by the given scalar,
    /// leaving the azimuth and latitude components unchanged.
    ///
    /// # Arguments
    ///
    /// * `self` - A mutable reference to the Spherical vector to be multiplied.
    /// * `rhs` - The scalar value to multiply by.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::MulAssign;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let mut vector = SphericalBuilder::with(1.0, 2.0, 3.0).build();
    /// vector *= 2.0;
    ///
    /// assert_eq!(vector.radius(), 2.0);
    /// assert_eq!(vector.azimuth(), 2.0);
    /// assert_eq!(vector.latitude(), 3.0);
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0[Spherical::RADIUS_IDX] *= rhs;
    }
}

/// Implements scalar division for Spherical vectors.
impl Div<f64> for Vec3d<Spherical> {
    /// The type of the result of the division by a floating-point scalar operation.
    type Output = Self;

    /// Divides the Spherical vector by a scalar.
    ///
    /// This function returns a new Spherical vector where the radius component is divided by the given scalar,
    /// while leaving the azimuth and latitude components unchanged.
    ///
    /// # Arguments
    ///
    /// * `self` - The Spherical vector to be divided.
    /// * `rhs` - The scalar value to divide by.
    ///
    /// # Returns
    ///
    /// A new Spherical vector with the radius component divided by the scalar, and other components unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Div;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let vector = SphericalBuilder::with(2.0, 2.0, 3.0).build();
    /// let divided_vector = vector / 2.0;
    ///
    /// assert_eq!(divided_vector.radius(), 1.0);
    /// assert_eq!(divided_vector.azimuth(), 2.0);
    /// assert_eq!(divided_vector.latitude(), 3.0);
    /// ```
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self(
            [
                self.0[Spherical::RADIUS_IDX] / rhs,
                self.0[Spherical::AZIMUTH_IDX],
                self.0[Spherical::LATITUDE_IDX],
            ],
            PhantomData::<Spherical> {},
        )
    }
}

/// Implements in-place scalar division for Spherical vectors.
impl DivAssign<f64> for Vec3d<Spherical> {
    /// Divides the Spherical vector in-place by a scalar.
    ///
    /// This function divides the radius component of the Spherical vector by the given scalar,
    /// leaving the azimuth and latitude components unchanged.
    ///
    /// # Arguments
    ///
    /// * `self` - A mutable reference to the Spherical vector to be divided.
    /// * `rhs` - The scalar value to divide by.
    ///
    /// # Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::DivAssign;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let mut vector = SphericalBuilder::with(2.0, 2.0, 3.0).build();
    /// vector /= 2.0;
    ///
    /// assert_eq!(vector.radius(), 1.0);
    /// assert_eq!(vector.azimuth(), 2.0);
    /// assert_eq!(vector.latitude(), 3.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.0[Spherical::RADIUS_IDX] /= rhs;
    }
}

/// Implements the `Debug` trait for Spherical vectors.
impl fmt::Debug for Vec3d<Spherical> {
    /// Formats the Spherical vector using the `Debug` trait.
    ///
    /// This function formats the Spherical vector as a debug string containing the radius, azimuth, and latitude components.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to a formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating success or failure in formatting the Spherical vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let vector = SphericalBuilder::with(3.0, 2.0, 1.0).build();
    /// let s = format!("{:?}", vector);
    /// assert_eq!(s.as_str(), "Vec3d { radius: 3.0, azimuth: 2.0, latitude: 1.0 }");
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vec3d")
            .field("radius", &self.0[Spherical::RADIUS_IDX])
            .field("azimuth", &self.0[Spherical::AZIMUTH_IDX])
            .field("latitude", &self.0[Spherical::LATITUDE_IDX])
            .finish()
    }
}

/// Trait representing a spatial direction in spherical coordinates.
///
/// The `SpatialDirection` trait defines methods to access the longitude and latitude components
/// of a position represented in spherical coordinates.
pub trait SpatialDirection {
    /// Returns the longitude component of the spatial direction.
    fn longitude(&self) -> f64;

    /// Returns the latitude component of the spatial direction.
    fn latitude(&self) -> f64;
}

/// Builder struct for creating instances of vectors in the spherical coordinate system.
///
/// The `SphericalBuilder` struct facilitates the construction of vectors in the spherical
/// coordinate system with various methods to set specific coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalBuilder {
    /// Radius component of the spherical vector.
    radius: f64,

    /// Azimuth component of the spherical vector.
    azimuth: f64,

    /// Latitude component of the spherical vector.
    latitude: f64,
}

impl SphericalBuilder {
    /// Creates a new `SphericalBuilder` instance with default values (0.0 for each component).
    #[inline]
    pub fn new() -> Self {
        Self {
            radius: 0.0,
            azimuth: 0.0,
            latitude: 0.0,
        }
    }

    /// Creates a `SphericalBuilder` instance with specified coordinates.
    #[inline]
    pub fn with(radius: f64, azimuth: f64, latitude: f64) -> Self {
        Self {
            radius,
            azimuth,
            latitude,
        }
    }

    /// Constructs a spherical position with the given radius and direction.
    #[inline]
    pub fn make<S: SpatialDirection>(radius: f64, direction: &S) -> Self {
        Self::with(radius, direction.longitude(), direction.latitude())
    }

    /// Creates a unit `SphericalBuilder` instance with the specified azimuth and latitude.
    #[inline]
    pub fn unit(azimuth: f64, latitude: f64) -> Self {
        Self {
            radius: 1.0,
            azimuth,
            latitude,
        }
    }

    /// Sets the radius component and returns a builder for chaining.
    #[inline]
    pub fn radius(mut self, value: f64) -> Self {
        self.radius = value;
        self
    }

    /// Sets the azimuth component and returns a builder for chaining.
    #[inline]
    pub fn azimuth(mut self, value: f64) -> Self {
        self.azimuth = value;
        self
    }

    /// Sets the latitude component and returns a builder for chaining.
    #[inline]
    pub fn latitude(mut self, value: f64) -> Self {
        self.latitude = value;
        self
    }

    /// Builds and returns a `Vec3d<Spherical>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Spherical> {
        Vec3d::<Spherical>(
            [self.radius, self.azimuth, self.latitude],
            PhantomData::<Spherical> {},
        )
    }
}

impl Default for SphericalBuilder {
    /// Creates a default `SphericalBuilder` instance with default values (0.0 for each component).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<S: SpatialDirection> From<&S> for SphericalBuilder {
    /// Constructs a unit spherical position with the given position.
    #[inline]
    fn from(direction: &S) -> Self {
        Self::unit(direction.longitude(), direction.latitude())
    }
}

/// Implements conversion from an iterator over `f64` values into a Spherical vector builder.
impl FromIterator<f64> for SphericalBuilder {
    /// Constructs a Spherical vector builder from an iterator over `f64` values.
    ///
    /// This function consumes the iterator and constructs a Spherical vector builder from the first three `f64` values
    /// encountered in the iterator. If the iterator contains fewer than three values, the remaining components are set to zero.
    /// If the iterator contains more than three values, only the first three values are used to construct the vector.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator over `f64` values.
    ///
    /// # Returns
    ///
    /// A Spherical vector builder constructed from the first three `f64` values encountered in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::iter::FromIterator;
    /// use ephem::core::vectors::{Spherical, SphericalBuilder, Vec3d};
    ///
    /// let iter = vec![3.0, 2.0, 1.0, 0.0].into_iter();
    /// let builder = SphericalBuilder::from_iter(iter);
    /// let vector = builder.build();
    ///
    /// assert_eq!(vector.radius(), 3.0);
    /// assert_eq!(vector.azimuth(), 2.0);
    /// assert_eq!(vector.latitude(), 1.0);
    /// ```
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut radius = 0.0;
        let mut azimuth = 0.0;
        let mut latitude = 0.0;

        for (idx, value) in iter.into_iter().enumerate() {
            match idx {
                Spherical::RADIUS_IDX => radius = value,
                Spherical::AZIMUTH_IDX => azimuth = value,
                Spherical::LATITUDE_IDX => latitude = value,
                _ => continue,
            }
        }

        Self::with(radius, azimuth, latitude)
    }
}
