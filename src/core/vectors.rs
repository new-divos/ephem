use std::{
    convert::From,
    f64::consts::PI,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{Canonizable, CrossMul, DotMul, FloatExt, Normalizable};
use crate::core::consts::PI2;

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

pub trait ToCartesian {
    fn to_c(&self) -> Vec3d<Cartesian>;
}

pub trait ToCylindrical {
    fn to_y(&self) -> Vec3d<Cylindrical>;
}

pub trait ToSpherical {
    fn to_s(&self) -> Vec3d<Spherical>;
}

/// Three-dimensional vector struct parameterized by a coordinate system.
///
/// The `Vec3d` struct represents a three-dimensional vector with coordinates stored as an array of
/// three `f64` values. The choice of coordinate system is determined by the type parameter `S`, which
/// must implement the `CoordinateSystem` trait.
#[derive(Clone, Debug, PartialEq)]
pub struct Vec3d<S: CoordinateSystem>(
    /// Array representing the three coordinates of the vector.
    [f64; 3],
    /// PhantomData marker to tie the coordinate system type to the vector.
    PhantomData<S>,
);

impl<S> Vec3d<S>
where
    S: CoordinateSystem,
    Vec3d<S>: Div<f64, Output = Self>,
{
    #[inline]
    pub fn checked_div(self, rhs: f64) -> Option<Self> {
        if rhs != 0.0 {
            Some(self / rhs)
        } else {
            None
        }
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

impl<S> Mul<Vec3d<S>> for f64
where
    S: CoordinateSystem,
    Vec3d<S>: Mul<f64, Output = Self>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Vec3d<S>) -> Self::Output {
        rhs.mul(self)
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

impl ToCylindrical for Vec3d<Cartesian> {
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

impl ToSpherical for Vec3d<Cartesian> {
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

impl<S> From<Vec3d<S>> for Vec3d<Cartesian>
where
    S: CoordinateSystem,
    Vec3d<S>: ToCartesian,
{
    #[inline]
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_c()
    }
}

impl Neg for Vec3d<Cartesian> {
    type Output = Self;

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

impl Add for Vec3d<Cartesian> {
    type Output = Self;

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

impl AddAssign for Vec3d<Cartesian> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[Cartesian::X_IDX] += rhs.0[Cartesian::X_IDX];
        self.0[Cartesian::Y_IDX] += rhs.0[Cartesian::Y_IDX];
        self.0[Cartesian::Z_IDX] += rhs.0[Cartesian::Z_IDX];
    }
}

impl Sub for Vec3d<Cartesian> {
    type Output = Self;

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

impl SubAssign for Vec3d<Cartesian> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[Cartesian::X_IDX] -= rhs.0[Cartesian::X_IDX];
        self.0[Cartesian::Y_IDX] -= rhs.0[Cartesian::Y_IDX];
        self.0[Cartesian::Z_IDX] -= rhs.0[Cartesian::Z_IDX];
    }
}

impl Mul<f64> for Vec3d<Cartesian> {
    type Output = Self;

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

impl MulAssign<f64> for Vec3d<Cartesian> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0[Cartesian::X_IDX] *= rhs;
        self.0[Cartesian::Y_IDX] *= rhs;
        self.0[Cartesian::Z_IDX] *= rhs;
    }
}

impl Div<f64> for Vec3d<Cartesian> {
    type Output = Self;

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

impl DivAssign<f64> for Vec3d<Cartesian> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.0[Cartesian::X_IDX] /= rhs;
        self.0[Cartesian::Y_IDX] /= rhs;
        self.0[Cartesian::Z_IDX] /= rhs;
    }
}

impl DotMul for Vec3d<Cartesian> {
    type Output = f64;

    #[inline]
    fn dot(self, rhs: Self) -> Self::Output {
        self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1] + self.0[2] * rhs.0[2]
    }
}

impl CrossMul for Vec3d<Cartesian> {
    type Output = Self;

    #[inline]
    fn cross(self, rhs: Self) -> Self::Output {
        Vec3d::<Cartesian>(
            [
                self.0[Cartesian::Z_IDX] * rhs.0[Cartesian::Y_IDX]
                    - self.0[Cartesian::Y_IDX] * rhs.0[Cartesian::Z_IDX],
                self.0[Cartesian::X_IDX] * rhs.0[Cartesian::Z_IDX]
                    - self.0[Cartesian::Z_IDX] * rhs.0[Cartesian::X_IDX],
                self.0[Cartesian::Y_IDX] * rhs.0[Cartesian::X_IDX]
                    - self.0[Cartesian::X_IDX] * rhs.0[Cartesian::Y_IDX],
            ],
            PhantomData::<Cartesian> {},
        )
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

impl Canonizable for Vec3d<Cylindrical> {
    fn canonic(self) -> Self {
        let mut radius = self.0[Cylindrical::RADIUS_IDX];
        let mut azimuth = self.0[Cylindrical::AZIMUTH_IDX];

        if radius < 0.0 {
            radius = -radius;
            azimuth += PI;
        }

        Vec3d::<Cylindrical>(
            [radius, azimuth.fmod(PI2), self.0[Cylindrical::ALTITUDE_IDX]],
            PhantomData::<Cylindrical> {},
        )
    }
}

impl ToCartesian for Vec3d<Cylindrical> {
    fn to_c(&self) -> Vec3d<Cartesian> {
        let (sa, ca) = self.0[Cylindrical::AZIMUTH_IDX].sin_cos();
        let rho = self.0[Cylindrical::RADIUS_IDX];

        Vec3d::<Cartesian>(
            [ca * rho, sa * rho, self.0[Cylindrical::ALTITUDE_IDX]],
            PhantomData::<Cartesian> {},
        )
    }
}

impl ToSpherical for Vec3d<Cylindrical> {
    fn to_s(&self) -> Vec3d<Spherical> {
        let rho = self.0[Cylindrical::RADIUS_IDX];
        let z = self.0[Cylindrical::ALTITUDE_IDX];

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

impl<S> From<Vec3d<S>> for Vec3d<Cylindrical>
where
    S: CoordinateSystem,
    Vec3d<S>: ToCylindrical,
{
    #[inline]
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_y()
    }
}

impl Neg for Vec3d<Cylindrical> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Vec3d::<Cylindrical>(
            [
                -self.0[Cylindrical::RADIUS_IDX],
                self.0[Cylindrical::AZIMUTH_IDX] + PI,
                -self.0[Cylindrical::ALTITUDE_IDX],
            ],
            PhantomData::<Cylindrical> {},
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

impl ToCartesian for Vec3d<Spherical> {
    fn to_c(&self) -> Vec3d<Cartesian> {
        let (sa, ca) = self.0[Spherical::AZIMUTH_IDX].sin_cos();
        let (sl, cl) = self.0[Spherical::LATITUDE_IDX].sin_cos();
        let r = self.0[Spherical::RADIUS_IDX];
        let rho = r * cl;

        Vec3d::<Cartesian>([ca * rho, sa * rho, r * sl], PhantomData::<Cartesian> {})
    }
}

impl ToCylindrical for Vec3d<Spherical> {
    fn to_y(&self) -> Vec3d<Cylindrical> {
        let (sl, cl) = self.0[Spherical::LATITUDE_IDX].sin_cos();
        let r = self.0[Spherical::RADIUS_IDX];

        Vec3d::<Cylindrical>(
            [cl * r, self.0[Spherical::AZIMUTH_IDX], sl * r],
            PhantomData::<Cylindrical> {},
        )
    }
}

impl<S> From<Vec3d<S>> for Vec3d<Spherical>
where
    S: CoordinateSystem,
    Vec3d<S>: ToSpherical,
{
    fn from(vector: Vec3d<S>) -> Self {
        vector.to_s()
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
