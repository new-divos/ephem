#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI};

use approx::assert_relative_eq;

use rand::Rng;

use ephem::core::vectors::*;
use ephem::core::{consts::PI2, DotMul};

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
fn vec3d_conversion_into_tuple_test() {
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

/// Test function for converting `Vec3d` vectors between different coordinate systems.
///
/// This test function generates random vectors in Cartesian, cylindrical, and spherical coordinate systems,
/// converts them to other coordinate systems, and then converts them back to the original coordinate system.
/// It ensures that the conversion functions maintain consistency by comparing the converted vectors with the original ones.
#[test]
fn vec3d_converion_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let c = CartesianBuilder::with(x, y, z).build();

        // Convert to cylindrical and back to Cartesian
        let y = c.to_y();
        let t: Vec3d<Cartesian> = y.into();

        // Compare components
        assert_relative_eq!(
            t.x(),
            c.x(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.y(),
            c.y(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.z(),
            c.z(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );

        // Convert to spherical and back to Cartesian
        let s = c.to_s();
        let t: Vec3d<Cartesian> = s.into();

        // Compare components
        assert_relative_eq!(
            t.x(),
            c.x(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.y(),
            c.y(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.z(),
            c.z(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
    }

    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = gen_cylindrical(&mut rng);
        let y = CylindricalBuilder::with(radius.abs(), azimuth, altitude).build();

        // Convert to cartesian and back to Cylindrical
        let c = y.to_c();
        let t: Vec3d<Cylindrical> = c.into();

        // Compare components
        assert_relative_eq!(
            t.radius(),
            y.radius(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.azimuth(),
            y.azimuth(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.altitude(),
            y.altitude(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );

        // Convert to spherical and back to Cylindrical
        let s = y.to_s();
        let t: Vec3d<Cylindrical> = s.into();

        // Compare components
        assert_relative_eq!(
            t.radius(),
            y.radius(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.azimuth(),
            y.azimuth(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.altitude(),
            y.altitude(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
    }

    for _ in 0..shared::ITERATIONS {
        // Generate random Spherical vector
        let (radius, azimuth, latitude) = gen_spherical(&mut rng);
        let s = SphericalBuilder::with(radius.abs(), azimuth, latitude).build();

        // Convert to cylindrical and back to Spherical
        let y = s.to_y();
        let t: Vec3d<Spherical> = y.into();

        // Compare components
        assert_relative_eq!(
            t.radius(),
            s.radius(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.azimuth(),
            s.azimuth(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.latitude(),
            s.latitude(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );

        // Convert to cartesian and back to Spherical
        let c = s.to_c();
        let t: Vec3d<Spherical> = c.into();

        // Compare components
        assert_relative_eq!(
            t.radius(),
            s.radius(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.azimuth(),
            s.azimuth(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
        assert_relative_eq!(
            t.latitude(),
            s.latitude(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
    }
}

/// Test function for negating a Cartesian `Vec3d` vector.
///
/// This test function verifies the correctness of the negation operation on Cartesian `Vec3d` vectors.
/// It generates random Cartesian vectors and checks whether the negation operation produces the expected result.
#[test]
fn cartesion_vec3d_negation_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Perform negation
        let n = -v;

        // Verify components of the resulting vector
        assert_eq!(n.x(), -x);
        assert_eq!(n.y(), -y);
        assert_eq!(n.z(), -z);
    }
}

/// Test function for addition of two `Vec3d` vectors.
///
/// This test function generates random Cartesian vectors, performs addition between them,
/// and verifies that the components of the resulting vector are the sum of the corresponding components of the input vectors.
#[test]
fn cartesian_vec3d_addition_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform addition
        let r = v1 + v2;

        // Verify components of the resulting vector
        assert_eq!(r.x(), x1 + x2);
        assert_eq!(r.y(), y1 + y2);
        assert_eq!(r.z(), z1 + z2);
    }
}

/// Test function for in-place addition of Cartesian `Vec3d` vectors.
///
/// This test function verifies the correctness of the in-place addition operation on Cartesian `Vec3d` vectors.
/// It generates random Cartesian vectors, performs in-place addition with another random vector,
/// and checks whether the resulting vector has the expected components.
#[test]
fn cartesian_vec3d_addition_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = gen_cartesian(&mut rng);
        let mut v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform addition with assignment
        v1 += v2;

        // Verify components of the resulting vector
        assert_eq!(v1.x(), x1 + x2);
        assert_eq!(v1.y(), y1 + y2);
        assert_eq!(v1.z(), z1 + z2);
    }
}

/// Test function for subtraction of two `Vec3d` vectors.
///
/// This test function generates random Cartesian vectors, performs subtraction between them,
/// and verifies that the components of the resulting vector are the difference of the corresponding components
/// of the input vectors.
#[test]
fn cartesian_vec3d_substraction_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform subtraction
        let r = v1 - v2;

        // Verify components of the resulting vector
        assert_eq!(r.x(), x1 - x2);
        assert_eq!(r.y(), y1 - y2);
        assert_eq!(r.z(), z1 - z2);
    }
}

/// Test function for performing subtraction with assignment on Cartesian `Vec3d` vectors.
///
/// This function tests the subtraction with assignment operation (`-=`) on Cartesian `Vec3d` vectors.
/// It generates random Cartesian vectors, performs subtraction with assignment, and then verifies
/// that the components of the resulting vector are correct.
#[test]
fn cartesian_vec3d_substraction_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = gen_cartesian(&mut rng);
        let mut v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform subtraction with assignment
        v1 -= v2;

        // Verify components of the resulting vector
        assert_eq!(v1.x(), x1 - x2);
        assert_eq!(v1.y(), y1 - y2);
        assert_eq!(v1.z(), z1 - z2);
    }
}

/// Test function for Cartesian vector multiplication by scalar.
///
/// This test function verifies the correctness of the scalar multiplication operation
/// for Cartesian vectors. It generates random Cartesian vectors and a random scalar,
/// then performs scalar multiplication by both left-hand side and right-hand side
/// operations. The resulting vectors are compared against expected values.
#[test]
fn cartesian_vec3d_multiplication_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;

        // Perform multiplication by the scalar
        let r = v.clone() * scalar;

        // Verify components of the resulting vector
        assert_eq!(r.x(), x * scalar);
        assert_eq!(r.y(), y * scalar);
        assert_eq!(r.z(), z * scalar);

        // Perform multiplication by the scalar
        let r = scalar * v;

        // Verify components of the resulting vector
        assert_eq!(r.x(), scalar * x);
        assert_eq!(r.y(), scalar * y);
        assert_eq!(r.z(), scalar * z);
    }
}

/// Test function for multiplying `Vec3d<Cartesian>` vectors by a scalar with assignment.
///
/// This test function verifies the correct behavior of the multiplication operation with assignment
/// (`*=`) between a `Vec3d<Cartesian>` vector and a scalar.
///
/// It generates random Cartesian vectors and a random scalar value, performs the multiplication
/// operation with assignment, and then verifies if the resulting vector has its components correctly
/// multiplied by the scalar.
#[test]
fn cartesian_vec3d_multiplication_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let mut v = CartesianBuilder::with(x, y, z).build();
        
        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        
        // Perform multiplication by the scalar with assignment
        v *= scalar;

        // Verify components of the resulting vector
        assert_eq!(v.x(), x * scalar);
        assert_eq!(v.y(), y * scalar);
        assert_eq!(v.z(), z * scalar);
    }
}


/// Test function to verify division operation of a 3D Cartesian vector by a scalar.
/// 
/// This function generates random Cartesian vectors and floating-point scalars, 
/// and then performs division operation on the vector by the scalar. 
/// It verifies the correctness of division operation by comparing the components 
/// of the resulting vector with the expected values obtained by dividing the 
/// original vector components by the scalar.
#[test]
fn cartesian_vec3d_divison_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        if scalar.abs() < shared::EPSILON {
            continue;
        }

        // Perform division by the scalar
        let r = v / scalar;

        // Verify components of the resulting vector
        assert_eq!(r.x(), x / scalar);
        assert_eq!(r.y(), y / scalar);
        assert_eq!(r.z(), z / scalar);
    }
}

/// Test function to verify division with assignment operation of a 3D Cartesian vector by a scalar.
/// 
/// This function generates random Cartesian vectors and floating-point scalars, 
/// and then performs division with assignment operation on the vector by the scalar. 
/// It verifies the correctness of division operation by comparing the components 
/// of the resulting vector with the expected values obtained by dividing the 
/// original vector components by the scalar.
#[test]
fn cartesian_vec3d_divison_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = gen_cartesian(&mut rng);
        let mut v = CartesianBuilder::with(x, y, z).build();
        
        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        if scalar.abs() < shared::EPSILON {
            continue;
        }
        
        // Perform multiplication by the scalar with assignment
        v /= scalar;

        // Verify components of the resulting vector
        assert_eq!(v.x(), x / scalar);
        assert_eq!(v.y(), y / scalar);
        assert_eq!(v.z(), z / scalar);
    }
}

/// Test function for dot product multiplication of two `Vec3d` vectors.
///
/// This test function generates random Cartesian vectors, performs dot product multiplication between them,
/// and verifies that the result is equal to the dot product of the corresponding components of the input vectors.
#[test]
fn cartesian_vec3d_dot_multiplication_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform dot product multiplication
        let r = v1.dot(v2);

        // Verify result
        assert_eq!(r, x1 * x2 + y1 * y2 + z1 * z2);
    }
}
