use std::convert::Into;
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
pub trait CoordinateSystem {}

/// Three-dimensional vector struct parameterized by a coordinate system.
///
/// The `Vec3d` struct represents a three-dimensional vector with coordinates stored as an array of
/// three `f64` values. The choice of coordinate system is determined by the type parameter `S`, which
/// must implement the `CoordinateSystem` trait.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3d<S: CoordinateSystem> (
    /// Array representing the three coordinates of the vector.
    [f64; 3],

    /// PhantomData marker to tie the coordinate system type to the vector.
    PhantomData<S>
);

/// Struct representing the Cartesian coordinate system.
pub struct Cartesian;

/// The `Cartesian` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait. 
impl CoordinateSystem for Cartesian {}

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
        (
            self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]
        ).sqrt()
    }
}

/// Implementation of the `Into` trait for converting a `Vec3d<Cartesian>` into a tuple.
///
/// This allows seamless conversion of a Cartesian vector into a tuple of its components.
#[allow(clippy::from_over_into)]
impl Into<(f64, f64, f64)> for Vec3d<Cartesian> {
    /// Converts the three-dimensional vector into a tuple of its components.
    ///
    /// # Returns
    ///
    /// A tuple representing the X, Y, and Z components of the vector.
    #[inline]
    fn into(self) -> (f64, f64, f64) {
        (self.0[Cartesian::X_IDX], self.0[Cartesian::Y_IDX], self.0[Cartesian::Z_IDX])
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
    z: f64
}

impl CartesianBuilder {
    /// Creates a new builder instance with default values (0.0 for each coordinate).
    #[inline]
    pub fn new() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Creates a builder instance with specified cartesian coordinates.
    #[inline]
    pub fn with(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Creates a builder instance representing the unit vector along the X-axis.
    #[inline]
    pub fn unit_x() -> Self {
        Self { x: 1.0, y: 0.0, z: 0.0 }
    }

    /// Creates a builder instance representing the unit vector along the Y-axis.
    #[inline]
    pub fn unit_y() -> Self {
        Self { x: 0.0, y: 1.0, z: 0.0 }
    }

    /// Creates a builder instance representing the unit vector along the Z-axis.
    #[inline]
    pub fn unit_z() -> Self {
        Self { x: 0.0, y: 0.0, z: 1.0 }
    }

    /// Sets the X-coordinate and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn x(&mut self, value: f64) -> &mut Self {
        self.x = value;
        self
    }

    /// Sets the Y-coordinate and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn y(&mut self, value: f64) -> &mut Self {
        self.y = value;
        self
    }

    /// Sets the Z-coordinate and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn z(&mut self, value: f64) -> &mut Self {
        self.z = value;
        self
    }

    /// Builds and returns a `Vec3d<Cartesian>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Cartesian> {
        Vec3d::<Cartesian>(
            [self.x, self.y, self.z],
            PhantomData::<Cartesian> {}
        )
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

/// The `Cylindrical` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait. 
impl CoordinateSystem for Cylindrical {}

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
        (
            self.0[Cylindrical::RADIUS_IDX] * self.0[Cylindrical::RADIUS_IDX] +
            self.0[Cylindrical::ALTITUDE_IDX] * self.0[Cylindrical::ALTITUDE_IDX]
        ).sqrt()
    }
}

/// Implementation of the `Into` trait for converting a `Vec3d<Cylindrical>` into a tuple.
///
/// This allows seamless conversion of a cylindrical vector into a tuple of its components.
#[allow(clippy::from_over_into)]
impl Into<(f64, f64, f64)> for Vec3d<Cylindrical> {
    /// Converts the three-dimensional vector into a tuple of its components.
    ///
    /// # Returns
    ///
    /// A tuple representing the radius, azimuth, and altitude components of the vector.
    #[inline]
    fn into(self) -> (f64, f64, f64) {
        (
            self.0[Cylindrical::RADIUS_IDX],
            self.0[Cylindrical::AZIMUTH_IDX],
            self.0[Cylindrical::ALTITUDE_IDX]
        )
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
        Self { radius: 0.0, azimuth: 0.0, altitude: 0.0 }
    }

    /// Creates a `CylindricalBuilder` instance with specified coordinates.
    #[inline]
    pub fn with(radius: f64, azimuth: f64, altitude: f64) -> Self {
        Self { radius, azimuth, altitude }
    }

    /// Sets the radius component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn radius(&mut self, value: f64) -> &mut Self {
        self.radius = value;
        self
    }

    /// Sets the azimuth component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn azimuth(&mut self, value: f64) -> &mut Self {
        self.azimuth = value;
        self
    }

    /// Sets the altitude component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn altitude(&mut self, value: f64) -> &mut Self {
        self.altitude = value;
        self
    }

    /// Builds and returns a `Vec3d<Cylindrical>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Cylindrical> {
        Vec3d::<Cylindrical> (
            [self.radius, self.azimuth, self.altitude],
            PhantomData::<Cylindrical> {}
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

/// The `Spherical` struct implements the `CoordinateSystem` trait, indicating that it adheres
/// to the requirements set by the trait. 
impl CoordinateSystem for Spherical {}

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

/// Implementation of the `Into` trait for converting a `Vec3d<Spherical>` into a tuple.
///
/// This allows seamless conversion of a spherical vector into a tuple of its components.
#[allow(clippy::from_over_into)]
impl Into<(f64, f64, f64)> for Vec3d<Spherical> {
    /// Converts the three-dimensional vector into a tuple of its components.
    ///
    /// # Returns
    ///
    /// A tuple representing the radius, azimuth, and latitude components of the vector.
    #[inline]
    fn into(self) -> (f64, f64, f64) {
        (
            self.0[Spherical::RADIUS_IDX],
            self.0[Spherical::AZIMUTH_IDX],
            self.0[Spherical::LATITUDE_IDX]
        )
    }
}

/// Trait representing a position in spherical coordinates.
///
/// The `SphericalPosition` trait defines methods to access the azimuth and latitude components
/// of a position represented in spherical coordinates.
pub trait SphericalPosition {
    /// Returns the azimuth component of the position.
    fn azimuth(&self) -> f64;

    /// Returns the latitude component of the position.
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
    latitude: f64
}

impl SphericalBuilder {
    /// Creates a new `SphericalBuilder` instance with default values (0.0 for each component).
    #[inline]
    pub fn new() -> Self {
        Self { radius: 0.0, azimuth: 0.0, latitude: 0.0 }
    } 

    /// Creates a `SphericalBuilder` instance with specified coordinates.
    #[inline]
    pub fn with(radius: f64, azimuth: f64, latitude: f64) -> Self {
        Self { radius, azimuth, latitude }
    }

    /// Creates a `SphericalBuilder` instance with a specified radius and position implementing
    /// the `SphericalPosition` trait.
    #[inline]
    pub fn with_pos<P: SphericalPosition>(radius: f64, position: &P) -> Self {
        Self::with(radius, position.azimuth(), position.latitude())
    }

    /// Creates a unit `SphericalBuilder` instance with the specified azimuth and latitude.
    #[inline]
    pub fn unit(azimuth: f64, latitude: f64) -> Self {
        Self { radius: 1.0, azimuth, latitude }
    }

    /// Creates a unit `SphericalBuilder` instance using a position implementing the
    /// `SphericalPosition` trait.
    #[inline]
    pub fn unit_pos<P: SphericalPosition>(position: &P) -> Self {
        Self::unit(position.azimuth(), position.latitude())
    }

    /// Sets the radius component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn radius(&mut self, value: f64) -> &mut Self {
        self.radius = value;
        self
    }

    /// Sets the azimuth component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn azimuth(&mut self, value: f64) -> &mut Self {
        self.azimuth = value;
        self
    }

    /// Sets the latitude component and returns a mutable reference to the builder for chaining.
    #[inline]
    pub fn latitude(&mut self, value: f64) -> &mut Self {
        self.latitude = value;
        self
    }

    /// Builds and returns a `Vec3d<Spherical>` instance using the configured coordinates.
    #[inline]
    pub fn build(self) -> Vec3d<Spherical> {
        Vec3d::<Spherical> (
            [self.radius, self.azimuth, self.latitude],
            PhantomData::<Spherical> {}
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