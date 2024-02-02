use num_traits::float::Float;

pub mod consts;
pub mod error;
pub mod vectors;

pub type Result<T> = std::result::Result<T, crate::core::error::Error>;

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
