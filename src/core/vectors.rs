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
/// to the requirements set by the trait. It also includes constants defining the indices for
/// commonly used Cartesian coordinates (X, Y, and Z).
impl CoordinateSystem for Cartesian {}

impl Cartesian {
    /// Constant representing the X-axis index in Cartesian coordinates.
    const X_IDX: usize = 0;

    /// Constant representing the Y-axis index in Cartesian coordinates.
    const Y_IDX: usize = 1;

    /// Constant representing the Z-axis index in Cartesian coordinates.
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
    x: f64,
    y: f64,
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

impl CoordinateSystem for Cylindrical {}

impl Cylindrical {
    const RADIUS_IDX: usize = 0;
    const AZIMUTH_IDX: usize = 1;
    const ALTITUDE_IDX: usize = 2;
}

impl Vec3d<Cylindrical> {
    #[inline]
    pub fn radius(&self) -> f64 {
        self.0[Cylindrical::RADIUS_IDX]
    }

    #[inline]
    pub fn azimuth(&self) -> f64 {
        self.0[Cylindrical::AZIMUTH_IDX]
    }

    #[inline]
    pub fn altitude(&self) -> f64 {
        self.0[Cylindrical::ALTITUDE_IDX]
    }
}

impl Vec3dNorm for Vec3d<Cylindrical> {
    #[inline]
    fn norm(&self) -> f64 {
        (
            self.0[Cylindrical::RADIUS_IDX] * self.0[Cylindrical::RADIUS_IDX] +
            self.0[Cylindrical::ALTITUDE_IDX] * self.0[Cylindrical::ALTITUDE_IDX]
        ).sqrt()
    }
}

#[allow(clippy::from_over_into)]
impl Into<(f64, f64, f64)> for Vec3d<Cylindrical> {
    #[inline]
    fn into(self) -> (f64, f64, f64) {
        (
            self.0[Cylindrical::RADIUS_IDX],
            self.0[Cylindrical::AZIMUTH_IDX],
            self.0[Cylindrical::ALTITUDE_IDX]
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CylindricalBuilder {
    radius: f64,
    azimuth: f64,
    altitude: f64,
}

impl CylindricalBuilder {
    #[inline]
    pub fn new() -> Self {
        Self { radius: 0.0, azimuth: 0.0, altitude: 0.0 }
    }

    #[inline]
    pub fn with(radius: f64, azimuth: f64, altitude: f64) -> Self {
        Self { radius, azimuth, altitude }
    }

    #[inline]
    pub fn radius(&mut self, value: f64) -> &mut Self {
        self.radius = value;
        self
    }

    #[inline]
    pub fn azimuth(&mut self, value: f64) -> &mut Self {
        self.azimuth = value;
        self
    }

    #[inline]
    pub fn altitude(&mut self, value: f64) -> &mut Self {
        self.altitude = value;
        self
    }

    #[inline]
    pub fn build(self) -> Vec3d<Cylindrical> {
        Vec3d::<Cylindrical> (
            [self.radius, self.azimuth, self.altitude],
            PhantomData::<Cylindrical> {}
        )
    }
}

impl Default for CylindricalBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Struct representing the Spherical coordinate system.
pub struct Spherical;

impl CoordinateSystem for Spherical {}

impl Spherical {
    const RADIUS_IDX: usize = 0;
    const AZIMUTH_IDX: usize = 1;
    const LATITUDE_IDX: usize = 2;
}

impl Vec3d<Spherical> {
    #[inline]
    pub fn radius(&self) -> f64 {
        self.0[Spherical::RADIUS_IDX]
    }

    #[inline]
    pub fn azimuth(&self) -> f64 {
        self.0[Spherical::AZIMUTH_IDX]
    }

    #[inline]
    pub fn latitude(&self) -> f64 {
        self.0[Spherical::LATITUDE_IDX]
    }
}

impl Vec3dNorm for Vec3d<Spherical> {
    #[inline]
    fn norm(&self) -> f64 {
        self.0[Spherical::RADIUS_IDX].abs()
    }
}

#[allow(clippy::from_over_into)]
impl Into<(f64, f64, f64)> for Vec3d<Spherical> {
    #[inline]
    fn into(self) -> (f64, f64, f64) {
        (
            self.0[Spherical::RADIUS_IDX],
            self.0[Spherical::AZIMUTH_IDX],
            self.0[Spherical::LATITUDE_IDX]
        )
    }
}

pub trait SphericalPosition {
    fn azimuth(&self) -> f64;
    fn latitude(&self) -> f64;
}

#[derive(Clone, Debug, PartialEq)]
pub struct SphericalBuilder {
    radius: f64,
    azimuth: f64,
    latitude: f64
}

impl SphericalBuilder {
    #[inline]
    pub fn new() -> Self {
        Self { radius: 0.0, azimuth: 0.0, latitude: 0.0 }
    } 

    #[inline]
    pub fn with(radius: f64, azimuth: f64, latitude: f64) -> Self {
        Self { radius, azimuth, latitude }
    }

    #[inline]
    pub fn unit(azimuth: f64, latitude: f64) -> Self {
        Self { radius: 1.0, azimuth, latitude }
    }

    #[inline]
    pub fn unit_pos<P: SphericalPosition>(position: &P) -> Self {
        Self::unit(position.azimuth(), position.latitude())
    }

    #[inline]
    pub fn radius(&mut self, value: f64) -> &mut Self {
        self.radius = value;
        self
    }

    #[inline]
    pub fn azimuth(&mut self, value: f64) -> &mut Self {
        self.azimuth = value;
        self
    }

    #[inline]
    pub fn latitude(&mut self, value: f64) -> &mut Self {
        self.latitude = value;
        self
    }

    #[inline]
    pub fn build(self) -> Vec3d<Spherical> {
        Vec3d::<Spherical> (
            [self.radius, self.azimuth, self.latitude],
            PhantomData::<Spherical> {}
        )
    }
}

impl Default for SphericalBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}