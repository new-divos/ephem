use std::convert::From;
use std::marker::PhantomData;

/// Trait representing the ability to calculate the norm (magnitude) of a three-dimensional vector.
///
/// The `Vec3dNorm` trait defines a method for computing the norm of a vector in three-dimensional space.
/// Types implementing this trait are expected to provide a consistent and accurate calculation of
/// the vector's magnitude.
pub trait Vec3dNorm {
    fn norm(&self) -> f64;
}

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

/// Three-dimensional vector struct parameterized by a coordinate system.
///
/// The `Vec3d` struct represents a three-dimensional vector with coordinates stored as an array of
/// three `f64` values. The choice of coordinate system is determined by the type parameter `S`, which
/// must implement the `CoordinateSystem` trait.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3d<S: CoordinateSystem>(
    /// Array representing the three coordinates of the vector.
    [f64; 3],
    /// PhantomData marker to tie the coordinate system type to the vector.
    PhantomData<S>,
);

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

/// Struct representing the Cartesian coordinate system.
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

/// Implementation of the `Vec3dNorm` trait for three-dimensional vectors in the Cartesian
/// coordinate system.
///
/// This allows the calculation of the magnitude (norm) of a `Vec3d<Cartesian>` vector.
impl Vec3dNorm for Vec3d<Cartesian> {
    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]).sqrt()
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

/// Struct representing the Cylindrical coordinate system.
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

/// Implementation of the `Vec3dNorm` trait for three-dimensional vectors in the cylindrical
/// coordinate system.
///
/// This allows the calculation of the norm (magnitude) of a `Vec3d<Cylindrical>` vector.
impl Vec3dNorm for Vec3d<Cylindrical> {
    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> f64 {
        (self.0[Cylindrical::RADIUS_IDX] * self.0[Cylindrical::RADIUS_IDX]
            + self.0[Cylindrical::ALTITUDE_IDX] * self.0[Cylindrical::ALTITUDE_IDX])
            .sqrt()
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

/// Struct representing the Spherical coordinate system.
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

/// Implementation of the `Vec3dNorm` trait for three-dimensional vectors in the spherical
/// coordinate system.
///
/// This allows the calculation of the norm (magnitude) of a `Vec3d<Spherical>` vector.
impl Vec3dNorm for Vec3d<Spherical> {
    /// Computes and returns the magnitude of the three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The magnitude of the vector as a floating-point number.
    #[inline]
    fn norm(&self) -> f64 {
        self.0[Spherical::RADIUS_IDX].abs()
    }
}

/// Trait representing a spatial direction in spherical coordinates.
///
/// The `SpatialDirection` trait defines methods to access the azimuth and latitude components
/// of a position represented in spherical coordinates.
pub trait SpatialDirection {
    /// Returns the azimuth component of the spatial direction.
    fn azimuth(&self) -> f64;

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
        Self::with(radius, direction.azimuth(), direction.latitude())
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
        Self::unit(direction.azimuth(), direction.latitude())
    }
}
