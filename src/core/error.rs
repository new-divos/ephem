use std::fmt;

/// Enum representing various error types that may occur within the functionality of the module.
#[derive(Debug)]
pub enum Error {
    /// Error indicating a division by zero operation.
    ZeroDivisionError,

    /// Error indicating a failure in type conversion.
    ConversionError,

    /// Error indicating the occurrence of a singular matrix in a mathematical operation.
    SingularMatrixError,

    /// Error indicating an attribute-related issue, with additional information provided.
    ///
    /// # Parameters
    ///
    /// - `attr_name`: A static string providing information about the attribute causing the error.
    ///
    /// # Examples
    ///
    /// ```
    /// use ephem::core::error::Error;
    ///
    /// let attribute_error = Error::AttributeError("invalid_attribute");
    /// println!("Error: {:?}", attribute_error);
    /// ```
    AttributeError(&'static str),
}

/// Implementing the `std::error::Error` trait for the custom `Error` enum.
impl std::error::Error for Error {}

impl fmt::Display for Error {
    /// Implementing the `fmt::Display` trait for the custom `Error` enum.
    ///
    /// This allows instances of the `Error` enum to be formatted as strings, making it
    /// convenient for displaying user-friendly error messages.
    ///
    /// # Parameters
    ///
    /// - `f`: The formatter that will receive the formatted output.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating the success or failure of the formatting operation.
    ///
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::ZeroDivisionError => {
                write!(f, "Zero division error")
            }
            Self::ConversionError => {
                write!(f, "Cannot convert one value into another")
            }
            Self::SingularMatrixError => {
                write!(f, "Try to use the singular matrix")
            }
            Self::AttributeError(name) => {
                write!(f, "Illegal value for the attribute {name}")
            }
        }
    }
}

pub(crate) struct Fault {}

impl Fault {
    pub const UNCONV_MUL: &'static str = "Unconvertible multiplier";
    pub const UNCONV_DIV: &'static str = "Unconvertible divisor";
}
