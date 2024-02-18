#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI};

use rand::Rng;

use ephem::core::consts::PI2;
use ephem::core::vectors::*;

mod shared;

/// A test struct representing a spatial direction with azimuth and latitude coordinates.
///
/// This struct is primarily used for testing purposes and holds azimuth and latitude coordinates.
#[derive(Debug, Default)]
struct TestDirection {
    azimuth: f64,
    latitude: f64,
}

impl SpatialDirection for TestDirection {
    /// Returns the azimuth coordinate of the direction.
    #[inline]
    fn azimuth(&self) -> f64 {
        self.azimuth
    }

    /// Returns the latitude coordinate of the direction.
    #[inline]
    fn latitude(&self) -> f64 {
        self.latitude
    }
}

impl TestDirection {
    /// Creates a new `TestDirection` with random azimuth and latitude values.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait.
    #[inline]
    fn with<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            azimuth: PI2 * rng.gen::<f64>(),
            latitude: PI * rng.gen::<f64>() - FRAC_PI_2,
        }
    }
}

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
fn gen_cartesian<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
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
fn gen_cylindrical<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
        PI2 * rng.gen::<f64>(),
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
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
fn gen_spherical<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64, f64) {
    (
        (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE,
        PI2 * rng.gen::<f64>(),
        PI * rng.gen::<f64>() - FRAC_PI_2,
    )
}

/// Test case for creating Cartesian vectors using the CartesianBuilder.
///
/// This test verifies the correctness of creating Cartesian vectors with various methods provided
/// by the CartesianBuilder struct. It checks that vectors are constructed correctly with zero
/// values, unit values, specific values, and random values within a range.
///
/// # Panics
///
/// This test will panic if any of the assertions fail.
#[test]
fn create_cartesian_vec3d_test() {
    // Test building a vector with default values
    let z = CartesianBuilder::new().build();
    assert_eq!(z.x(), 0.0);
    assert_eq!(z.y(), 0.0);
    assert_eq!(z.z(), 0.0);

    // Test building a vector with default values using the `Default` trait
    let d = CartesianBuilder::default().build();
    assert_eq!(d.x(), 0.0);
    assert_eq!(d.y(), 0.0);
    assert_eq!(d.z(), 0.0);

    // Test building unit vectors along each axis
    let i = CartesianBuilder::unit_x().build();
    assert_eq!(i.x(), 1.0);
    assert_eq!(i.y(), 0.0);
    assert_eq!(i.z(), 0.0);

    let j = CartesianBuilder::unit_y().build();
    assert_eq!(j.x(), 0.0);
    assert_eq!(j.y(), 1.0);
    assert_eq!(j.z(), 0.0);

    let k = CartesianBuilder::unit_z().build();
    assert_eq!(k.x(), 0.0);
    assert_eq!(k.y(), 0.0);
    assert_eq!(k.z(), 1.0);

    // Test building vectors with random values
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let (x, y, z) = gen_cartesian(&mut rng);

        // Test building a vector with specified values
        let v1 = CartesianBuilder::with(x, y, z).build();
        assert_eq!(v1.x(), x);
        assert_eq!(v1.y(), y);
        assert_eq!(v1.z(), z);

        // Test building a vector with method chaining
        let v2 = CartesianBuilder::new().x(x).y(y).z(z).build();
        assert_eq!(v2.x(), x);
        assert_eq!(v2.y(), y);
        assert_eq!(v2.z(), z);
    }
}

/// Test function for creating cylindrical vectors.
///
/// This function tests the creation of cylindrical vectors using the `CylindricalBuilder`.
/// It verifies that vectors can be created with specific values or using default values.
/// Additionally, it tests random creation of vectors within specified ranges.
///
/// # Panics
///
/// This test will panic if any of the assertions fail.
#[test]
fn create_cylindrical_vec3d_test() {
    // Test building a vector with default values
    let z = CylindricalBuilder::new().build();
    assert_eq!(z.radius(), 0.0);
    assert_eq!(z.azimuth(), 0.0);
    assert_eq!(z.altitude(), 0.0);

    // Test building a vector with default values using the `Default` trait
    let d = CylindricalBuilder::default().build();
    assert_eq!(d.radius(), 0.0);
    assert_eq!(d.azimuth(), 0.0);
    assert_eq!(d.altitude(), 0.0);

    // Test building vectors with random values
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, altitude) = gen_cylindrical(&mut rng);

        // Test building a vector with specified values
        let v1 = CylindricalBuilder::with(radius, azimuth, altitude).build();
        assert_eq!(v1.radius(), radius);
        assert_eq!(v1.azimuth(), azimuth);
        assert_eq!(v1.altitude(), altitude);

        // Test building a vector with method chaining
        let v2 = CylindricalBuilder::new()
            .radius(radius)
            .azimuth(azimuth)
            .altitude(altitude)
            .build();
        assert_eq!(v2.radius(), radius);
        assert_eq!(v2.azimuth(), azimuth);
        assert_eq!(v2.altitude(), altitude);
    }
}

/// Test function for creating spherical vectors.
///
/// This function tests the creation of spherical vectors using the `SphericalBuilder`.
/// It verifies that vectors can be created with default values, random values, and specified values.
/// Additionally, it tests building unit vectors with specified values.
///
/// # Panics
///
/// This test will panic if any of the assertions fail.
#[test]
fn create_spherical_vec3d_test() {
    // Test building a vector with default values
    let z = SphericalBuilder::new().build();
    assert_eq!(z.radius(), 0.0);
    assert_eq!(z.azimuth(), 0.0);
    assert_eq!(z.latitude(), 0.0);

    // Test building a vector with default values using the `Default` trait
    let d = SphericalBuilder::default().build();
    assert_eq!(d.radius(), 0.0);
    assert_eq!(d.azimuth(), 0.0);
    assert_eq!(d.latitude(), 0.0);

    // Test building vectors with random values
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, latitude) = gen_spherical(&mut rng);

        // Test building a vector with specified values
        let v1 = SphericalBuilder::with(radius, azimuth, latitude).build();
        assert_eq!(v1.radius(), radius);
        assert_eq!(v1.azimuth(), azimuth);
        assert_eq!(v1.latitude(), latitude);

        // Test building a vector with method chaining
        let v2 = SphericalBuilder::new()
            .radius(radius)
            .azimuth(azimuth)
            .latitude(latitude)
            .build();
        assert_eq!(v2.radius(), radius);
        assert_eq!(v2.azimuth(), azimuth);
        assert_eq!(v2.latitude(), latitude);

        // Test building a unit vector with specified values
        let u = SphericalBuilder::unit(azimuth, latitude).build();
        assert_eq!(u.radius(), 1.0);
        assert_eq!(u.azimuth(), azimuth);
        assert_eq!(u.latitude(), latitude);
    }
}

/// Test function for creating spherical vectors with a given direction.
///
/// This function tests the creation of spherical vectors using a given direction provided by `TestDirection`.
/// It verifies that vectors can be created with random values for radius and direction.
/// Additionally, it tests building unit vectors from the given direction.
///
/// # Panics
///
/// This test will panic if any of the assertions fail.
#[test]
fn create_spherical_vec3d_with_position_test() {
    // Test building vectors with random values
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let radius = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() + shared::MIN_VALUE;
        let direction = TestDirection::with(&mut rng);

        // Test building a vector with specified radius and direction
        let v = SphericalBuilder::make(radius, &direction).build();
        assert_eq!(v.radius(), radius);
        assert_eq!(v.azimuth(), direction.azimuth());
        assert_eq!(v.latitude(), direction.latitude());

        // Test building a unit vector from the given direction
        let u = SphericalBuilder::from(&direction).build();
        assert_eq!(u.radius(), 1.0);
        assert_eq!(u.azimuth(), direction.azimuth());
        assert_eq!(u.latitude(), direction.latitude());
    }
}

/// Test function for converting vectors into tuples.
///
/// This function tests the conversion of vectors (`Vec3d`) into tuples of coordinates for various coordinate systems.
/// It generates vectors with random values and verifies that the conversion produces tuples with corresponding values.
///
/// # Panics
///
/// This test will panic if any of the assertions fail.
#[test]
fn convert_vec3d_into_tuple_test() {
    // Test converting Cartesian vectors into tuples
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let (x, y, z) = gen_cartesian(&mut rng);

        let v = CartesianBuilder::with(x, y, z).build();
        let (x_t, y_t, z_t) = v.into();

        assert_eq!(x_t, x);
        assert_eq!(y_t, y);
        assert_eq!(z_t, z);
    }

    // Test converting cylindrical vectors into tuples
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, altitude) = gen_cylindrical(&mut rng);

        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();
        let (radius_t, azimuth_t, altitude_t) = v.into();

        assert_eq!(radius_t, radius);
        assert_eq!(azimuth_t, azimuth);
        assert_eq!(altitude_t, altitude);
    }

    // Test converting spherical vectors into tuples
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, latitude) = gen_spherical(&mut rng);

        let v = SphericalBuilder::with(radius, azimuth, latitude).build();
        let (radius_t, azimuth_t, latitude_t) = v.into();

        assert_eq!(radius_t, radius);
        assert_eq!(azimuth_t, azimuth);
        assert_eq!(latitude_t, latitude);
    }
}
