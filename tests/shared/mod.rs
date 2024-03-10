use std::f64::consts::{FRAC_PI_2, PI};

use rand::Rng;

pub const ITERATIONS: i32 = 10000;
pub const MIN_VALUE: f64 = -10000.0;
pub const MAX_VALUE: f64 = 10000.0;

pub const EPSILON: f64 = 1e-8;

/// Generates random Cartesian coordinates within a specified range.
///
/// This function generates random Cartesian coordinates `(x, y, z)` within the range
/// defined by `shared::MIN_VALUE` and `shared::MAX_VALUE` using the provided random number generator `rng`.
///
/// # Arguments
///
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait.
///
/// # Returns
///
/// A tuple containing the generated Cartesian coordinates `(x, y, z)`.
#[inline]
pub fn gen_cartesian<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
    )
}

/// Generates random cylindrical coordinates within a specified range.
///
/// This function generates random cylindrical coordinates `(radius, azimuth, altitude)` within the range
/// defined by `shared::MIN_VALUE` and `shared::MAX_VALUE` for radius and altitude, and 0 to 2π for azimuth,
/// using the provided random number generator `rng`.
///
/// # Arguments
///
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait.
///
/// # Returns
///
/// A tuple containing the generated cylindrical coordinates `(radius, azimuth, altitude)`.
#[inline]
pub fn gen_cylindrical<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
        2.0 * PI * rng.gen::<f64>(),
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
    )
}

/// Generates random spherical coordinates within a specified range.
///
/// This function generates random spherical coordinates `(radius, azimuth, latitude)` within the range
/// defined by `shared::MIN_VALUE` and `shared::MAX_VALUE` for radius, 0 to 2π for azimuth,
/// and -π/2 to π/2 for latitude, using the provided random number generator `rng`.
///
/// # Arguments
///
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait.
///
/// # Returns
///
/// A tuple containing the generated spherical coordinates `(radius, azimuth, latitude)`.
#[inline]
pub fn gen_spherical<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (MAX_VALUE - MIN_VALUE) * rng.gen::<f64>() + MIN_VALUE,
        2.0 * PI * rng.gen::<f64>(),
        PI * rng.gen::<f64>() - FRAC_PI_2,
    )
}
