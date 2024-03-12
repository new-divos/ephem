use num_traits::float::Float;

pub mod consts;
pub mod error;
pub mod matrices;
pub mod vectors;

pub type Result<T> = std::result::Result<T, crate::core::error::Error>;

/// A trait for providing a canonical form of an object.
///
/// This trait defines a method `canonic` which transforms an object into its canonical form.
pub trait Canonizable {
    /// Transforms the object into its canonical form.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::Canonizable;
    ///
    /// struct MyStruct {
    ///     first: f64,
    ///     second: f64,
    /// }
    ///
    /// impl Canonizable for MyStruct {
    ///     fn canonic(self) -> Self {
    ///         MyStruct {
    ///             first: self.first.abs(),
    ///             second: self.second.abs(),
    ///         }
    ///     }
    /// }
    ///
    /// let my_struct = MyStruct { first: -1.0, second: 2.0 };
    /// let canonical_form = my_struct.canonic();
    /// assert_eq!(canonical_form.first, 1.0_f64);
    /// assert_eq!(canonical_form.second, 2.0_f64);
    /// ```
    fn canonic(self) -> Self;
}

/// A trait for providing the norm of an object.
///
/// This trait defines a method `norm` which calculates the norm of an object.
/// The associated type `Output` specifies the type of the result of the norm calculation.
pub trait Normalizable {
    /// The type of the result of the norm calculation.
    type Output;

    /// Calculates the norm of the object.
    ///
    /// # Returns
    ///
    /// The norm of the object.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ephem::core::Normalizable;
    ///
    /// struct MyVector {
    ///     x: f64,
    ///     y: f64,
    /// }
    ///
    /// impl Normalizable for MyVector {
    ///     type Output = f64;
    ///
    ///     fn norm(&self) -> Self::Output {
    ///         self.x.abs() + self.y.abs()
    ///     }
    /// }
    ///
    /// let vector = MyVector { x: 1.0, y: 2.0 };
    /// let norm = vector.norm();
    /// assert_eq!(norm, 3.0_f64);
    /// ```
    fn norm(&self) -> Self::Output;
}

/// A trait for performing dot multiplication between two objects.
///
/// This trait defines a method `dot` for performing dot multiplication between two objects of potentially different types.
/// The associated type `Output` specifies the type of the result of the dot multiplication operation.
pub trait DotMul<Rhs = Self> {
    /// The type of the result of the dot multiplication operation.
    type Output;

    /// Performs dot multiplication between two objects.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side object for dot multiplication.
    ///
    /// # Returns
    ///
    /// The result of the dot multiplication operation.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ephem::core::DotMul;
    ///
    /// struct MyVector {
    ///     x: f64,
    ///     y: f64,
    /// }
    ///
    /// impl DotMul<MyVector> for MyVector {
    ///     type Output = f64;
    ///
    ///     fn dot(self, rhs: MyVector) -> Self::Output {
    ///         self.x * rhs.x + self.y * rhs.y
    ///     }
    /// }
    ///
    /// let vector1 = MyVector { x: 1.0, y: 2.0 };
    /// let vector2 = MyVector { x: 3.0, y: 3.0 };
    /// let dot_product = vector1.dot(vector2);
    /// assert_eq!(dot_product, 9.0_f64);
    /// ```
    fn dot(self, rhs: Rhs) -> Self::Output;
}

/// A trait for performing cross multiplication between two objects.
///
/// This trait defines a method `cross` for performing cross multiplication between two objects of potentially different types.
/// The associated type `Output` specifies the type of the result of the cross multiplication operation.
pub trait CrossMul<Rhs = Self> {
    /// The type of the result of the cross multiplication operation.
    type Output;

    /// Performs cross multiplication between two objects.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side object for cross multiplication.
    ///
    /// # Returns
    ///
    /// The result of the cross multiplication operation.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ephem::core::CrossMul;
    ///
    /// struct MyVector {
    ///     x: f64,
    ///     y: f64,
    /// }
    ///
    /// impl CrossMul<MyVector> for MyVector {
    ///     type Output = MyVector;
    ///
    ///     fn cross(self, _rhs: MyVector) -> Self::Output {
    ///         MyVector { x: 0.0, y: 0.0 } // returning a placeholder value
    ///     }
    /// }
    ///
    /// let vector1 = MyVector { x: 1.0, y: 2.0 };
    /// let vector2 = MyVector { x: 3.0, y: 3.0 };
    /// let cross_product = vector1.cross(vector2);
    /// assert_eq!(cross_product.x, 0.0_f64);
    /// assert_eq!(cross_product.y, 0.0_f64);
    /// ```
    fn cross(self, rhs: Rhs) -> Self::Output;
}

/// A trait extending the functionality of floating-point numbers.
pub trait FloatExt {
    /// Returns the fractional part of the floating-point number.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::FloatExt;
    ///
    /// let x = 5.75;
    /// let fractional_part = x.frac();
    /// assert_eq!(fractional_part, 0.75);
    /// ```
    fn frac(self) -> Self;

    /// Calculates the remainder of the division of two floating-point numbers (`self % rhs`).
    ///
    /// # Parameters
    ///
    /// - `rhs`: The divisor.
    ///
    /// # Returns
    ///
    /// The remainder of the division operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::FloatExt;
    ///
    /// let x = 10.5;
    /// let y = 3.2;
    /// let remainder = x.fmod(y);
    /// assert!(((remainder - 0.9) as f64).abs() <= 0.001);
    /// ```
    fn fmod(self, rhs: Self) -> Self;
}

impl<T: Float> FloatExt for T {
    /// Returns the fractional part of the floating-point number.
    ///
    /// This method calculates the fractional part by subtracting the floored value
    /// from the original value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::FloatExt;
    ///
    /// let x = 5.75;
    /// let fractional_part = x.frac();
    /// assert_eq!(fractional_part, 0.75);
    /// ```
    #[inline]
    fn frac(self) -> Self {
        self - self.floor()
    }

    /// Calculates the remainder of the division of two floating-point numbers (`self % rhs`).
    ///
    /// This method computes the remainder by subtracting the product of the divisor and
    /// the floored division result from the dividend.
    ///
    /// # Parameters
    ///
    /// - `rhs`: The divisor.
    ///
    /// # Returns
    ///
    /// The remainder of the division operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::FloatExt;
    ///
    /// let x = 10.5;
    /// let y = 3.2;
    /// let remainder = x.fmod(y);
    /// assert!(((remainder - 0.9) as f64).abs() <= 0.001);
    /// ```
    #[inline]
    fn fmod(self, rhs: Self) -> Self {
        self - rhs * (self / rhs).floor()
    }
}
