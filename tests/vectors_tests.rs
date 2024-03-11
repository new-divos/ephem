#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI};

use approx::assert_relative_eq;

use rand::Rng;

use ephem::core::{consts::PI2, vectors::*, CrossMul, DotMul, Normalizable};

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

/// Test case for creating Cartesian vectors using the CartesianBuilder.
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
        let (x, y, z) = shared::gen_cartesian(&mut rng);

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
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);

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
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);

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
#[test]
fn vec3d_conversion_into_tuple_test() {
    // Test converting Cartesian vectors into tuples
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        let (x, y, z) = shared::gen_cartesian(&mut rng);

        let v = CartesianBuilder::with(x, y, z).build();
        let (x_t, y_t, z_t) = v.into();

        assert_eq!(x_t, x);
        assert_eq!(y_t, y);
        assert_eq!(z_t, z);
    }

    // Test converting cylindrical vectors into tuples
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);

        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();
        let (radius_t, azimuth_t, altitude_t) = v.into();

        assert_eq!(radius_t, radius);
        assert_eq!(azimuth_t, azimuth);
        assert_eq!(altitude_t, altitude);
    }

    // Test converting spherical vectors into tuples
    for _ in 0..shared::ITERATIONS {
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);

        let v = SphericalBuilder::with(radius, azimuth, latitude).build();
        let (radius_t, azimuth_t, latitude_t) = v.into();

        assert_eq!(radius_t, radius);
        assert_eq!(azimuth_t, azimuth);
        assert_eq!(latitude_t, latitude);
    }
}

/// Test function to verify the behavior of converting 3D vectors into iterators.
///
/// This test function generates random Cartesian, Cylindrical, and Spherical vectors and then converts them into iterators.
/// It ensures that iterating over the vectors via iterators produces the correct sequence of elements.
#[test]
fn vec3d_into_iterator_test() {
    let mut rng = rand::thread_rng();

    // Test iteration over Cartesian vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Iterate over the components and validate
        let mut iter = v.into_iter();
        assert_eq!(iter.next(), Some(x));
        assert_eq!(iter.next(), Some(y));
        assert_eq!(iter.next(), Some(z));
        assert_eq!(iter.next(), None);
    }

    // Test iteration over Cylindrical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Iterate over the components and validate
        let mut iter = v.into_iter();
        assert_eq!(iter.next(), Some(radius));
        assert_eq!(iter.next(), Some(azimuth));
        assert_eq!(iter.next(), Some(altitude));
        assert_eq!(iter.next(), None);
    }

    // Test iteration over Spherical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Spherical vector
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);
        let v = SphericalBuilder::with(radius, azimuth, latitude).build();

        // Iterate over the components and validate
        let mut iter = v.into_iter();
        assert_eq!(iter.next(), Some(radius));
        assert_eq!(iter.next(), Some(azimuth));
        assert_eq!(iter.next(), Some(latitude));
        assert_eq!(iter.next(), None);
    }
}

/// Test function to verify the behavior of iterators over different types of 3D vectors.
///
/// This test function generates random Cartesian, Cylindrical, and Spherical vectors and then iterates over their components.
/// It ensures that iterating over the vectors produces the correct sequence of elements, and also validates the conversion
/// of the iterator back into a vector.
#[test]
fn vec3d_iterator_test() {
    let mut rng = rand::thread_rng();

    // Test iteration over Cartesian vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Iterate over the components and validate
        for (idx, element) in v.iter().enumerate() {
            match idx {
                0 => assert_eq!(element, x),
                1 => assert_eq!(element, y),
                2 => assert_eq!(element, z),
                _ => continue,
            }
        }

        // Convert the iterator back into a vector and assert equality
        let mut t: Vec<f64> = Vec::new();
        for element in &v {
            t.push(element);
        }

        let u = CartesianBuilder::from_iter(t).build();
        assert_eq!(v, u);
    }

    // Test iteration over Cylindrical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Iterate over the components and validate
        for (idx, element) in v.iter().enumerate() {
            match idx {
                0 => assert_eq!(element, radius),
                1 => assert_eq!(element, azimuth),
                2 => assert_eq!(element, altitude),
                _ => continue,
            }
        }

        // Convert the iterator back into a vector and assert equality
        let mut t: Vec<f64> = Vec::new();
        for element in &v {
            t.push(element);
        }

        let u = CylindricalBuilder::from_iter(t).build();
        assert_eq!(v, u);
    }

    // Test iteration over Spherical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Spherical vector
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);
        let v = SphericalBuilder::with(radius, azimuth, latitude).build();

        // Iterate over the components and validate
        for (idx, element) in v.iter().enumerate() {
            match idx {
                0 => assert_eq!(element, radius),
                1 => assert_eq!(element, azimuth),
                2 => assert_eq!(element, latitude),
                _ => continue,
            }
        }

        // Convert the iterator back into a vector and assert equality
        let mut t: Vec<f64> = Vec::new();
        for element in &v {
            t.push(element);
        }

        let u = SphericalBuilder::from_iter(t).build();
        assert_eq!(v, u);
    }
}

/// Test function to verify the behavior of mutable iterators over different types of 3D vectors.
#[test]
fn vec3d_mutable_iterator_test() {
    let mut rng = rand::thread_rng();

    // Test iteration over Cartesian vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let mut v = CartesianBuilder::with(x, y, z).build();

        // Generate random coefficients
        let a = 2.0 * rng.gen::<f64>() - 1.0;
        let b = 2.0 * rng.gen::<f64>() - 1.0;
        let c = 2.0 * rng.gen::<f64>() - 1.0;

        // Modify components during iteration
        for (component, coefficient) in v.iter_mut().zip([a, b, c].iter()) {
            *component *= coefficient;
        }

        // Validate modified vector components
        assert_eq!(v.x(), a * x);
        assert_eq!(v.y(), b * y);
        assert_eq!(v.z(), c * z);

        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let mut v = CartesianBuilder::with(x, y, z).build();

        // Modify components during iteration
        for component in &mut v {
            *component = (*component).abs().sqrt();
        }

        // Validate modified vector components
        assert_eq!(v.x(), x.abs().sqrt());
        assert_eq!(v.y(), y.abs().sqrt());
        assert_eq!(v.z(), z.abs().sqrt());
    }

    // Test iteration over Cylindrical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let mut v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Generate random coefficients
        let a = 2.0 * rng.gen::<f64>() - 1.0;
        let b = PI2 * rng.gen::<f64>() - PI;
        let c = 2.0 * rng.gen::<f64>() - 1.0;

        // Modify components during iteration
        for (component, coefficient) in v.iter_mut().zip([a, b, c].iter()) {
            *component *= coefficient;
        }

        // Validate modified vector components
        assert_eq!(v.radius(), a * radius);
        assert_eq!(v.azimuth(), b * azimuth);
        assert_eq!(v.altitude(), c * altitude);

        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let mut v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Modify components during iteration
        for component in &mut v {
            *component = ((*component).powi(2) + 1.0).sqrt();
        }

        // Validate modified vector components
        assert_eq!(v.radius(), (radius.powi(2) + 1.0).sqrt());
        assert_eq!(v.azimuth(), (azimuth.powi(2) + 1.0).sqrt());
        assert_eq!(v.altitude(), (altitude.powi(2) + 1.0).sqrt());
    }

    // Test iteration over Spherical vectors
    for _ in 0..shared::ITERATIONS {
        // Generate random Spherical vector
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);
        let mut v = SphericalBuilder::with(radius, azimuth, latitude).build();

        // Generate random coefficients
        let a = 2.0 * rng.gen::<f64>() - 1.0;
        let b = PI2 * rng.gen::<f64>() - PI;
        let c = PI * rng.gen::<f64>() - FRAC_PI_2;

        // Modify components during iteration
        for (component, coefficient) in v.iter_mut().zip([a, b, c].iter()) {
            *component *= coefficient;
        }

        // Validate modified vector components
        assert_eq!(v.radius(), a * radius);
        assert_eq!(v.azimuth(), b * azimuth);
        assert_eq!(v.latitude(), c * latitude);

        // Generate random Spherical vector
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);
        let mut v = SphericalBuilder::with(radius, azimuth, latitude).build();

        // Modify components during iteration
        for component in &mut v {
            *component = ((*component).powi(2) + 2.0).sqrt();
        }

        // Validate modified vector components
        assert_eq!(v.radius(), (radius.powi(2) + 2.0).sqrt());
        assert_eq!(v.azimuth(), (azimuth.powi(2) + 2.0).sqrt());
        assert_eq!(v.latitude(), (latitude.powi(2) + 2.0).sqrt());
    }
}

/// Test function for converting `Vec3d` vectors between different coordinate systems.
#[test]
fn vec3d_converion_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
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
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
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
        let (radius, azimuth, latitude) = shared::gen_spherical(&mut rng);
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

/// Tests the calculation of the norm (magnitude) of Cartesian vectors.
#[test]
fn cartesian_vec3d_norma_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let v = CartesianBuilder::with(x, y, z).build();

        // Perform norma calculation
        let r = v.norm();

        // Verify result
        assert_relative_eq!(
            r,
            (v.x().powi(2) + v.y().powi(2) + v.z().powi(2)).sqrt(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
    }
}

/// Test function for negating a Cartesian `Vec3d` vector.
#[test]
fn cartesion_vec3d_negation_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_addition_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_addition_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let mut v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_substraction_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_substraction_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let mut v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_multiplication_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_multiplication_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_divison_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
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
#[test]
fn cartesian_vec3d_divison_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vector
        let (x, y, z) = shared::gen_cartesian(&mut rng);
        let mut v = CartesianBuilder::with(x, y, z).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        if scalar.abs() < shared::EPSILON {
            continue;
        }

        // Perform division by the scalar with assignment
        v /= scalar;

        // Verify components of the resulting vector
        assert_eq!(v.x(), x / scalar);
        assert_eq!(v.y(), y / scalar);
        assert_eq!(v.z(), z / scalar);
    }
}

/// Test function for dot product multiplication of two `Vec3d` vectors.
#[test]
fn cartesian_vec3d_dot_multiplication_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform dot product multiplication
        let r = v1.dot(v2);

        // Verify result
        assert_eq!(r, x1 * x2 + y1 * y2 + z1 * z2);
    }
}

/// Tests the cross product multiplication of Cartesian vectors.
#[test]
fn cartesian_vec3d_cross_multiplication_test() {
    let i = CartesianBuilder::unit_x().build();
    let j = CartesianBuilder::unit_y().build();
    let k = CartesianBuilder::unit_z().build();

    let r = i.clone().cross(j.clone());
    assert_eq!(r, k);

    let r = j.clone().cross(i.clone());
    assert_eq!(r, -k.clone());

    let r = j.clone().cross(k.clone());
    assert_eq!(r, i);

    let r = k.clone().cross(j.clone());
    assert_eq!(r, -i.clone());

    let r = k.clone().cross(i.clone());
    assert_eq!(r, j);

    let r = i.clone().cross(k.clone());
    assert_eq!(r, -j.clone());

    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let v1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let v2 = CartesianBuilder::with(x2, y2, z2).build();

        // Perform dot product multiplication
        let r = v1.cross(v2);

        // Verify result
        assert_eq!(r.x(), y1 * z2 - z1 * y2);
        assert_eq!(r.y(), z1 * x2 - x1 * z2);
        assert_eq!(r.z(), x1 * y2 - y1 * x2);
    }
}

/// Tests the calculation of the norm for vectors in cylindrical coordinates.
#[test]
fn cylindrical_vec3d_norma_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Perform norma calculation
        let r = v.norm();

        // Verify result
        assert_relative_eq!(
            r,
            (v.radius().powi(2) + v.altitude().powi(2)).sqrt(),
            epsilon = f64::EPSILON,
            max_relative = shared::EPSILON
        );
    }
}

/// Tests the negation operation for vectors in cylindrical coordinates.
#[test]
fn cylindrical_vec3d_negation_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Perform negation
        let n = -v;

        // Verify components of the resulting vector
        assert_eq!(n.radius(), -radius);
        assert_eq!(n.azimuth(), azimuth);
        assert_eq!(n.altitude(), -altitude);
    }
}

/// Tests the multiplication of a cylindrical vector by a scalar value.
#[test]
fn cylindrical_vec3d_multiplication_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;

        // Perform multiplication by the scalar
        let r = v.clone() * scalar;

        // Verify components of the resulting vector
        assert_eq!(r.radius(), radius * scalar);
        assert_eq!(r.azimuth(), azimuth);
        assert_eq!(r.altitude(), altitude * scalar);

        // Perform multiplication by the scalar
        let r = scalar * v;

        // Verify components of the resulting vector
        assert_eq!(r.radius(), scalar * radius);
        assert_eq!(r.azimuth(), azimuth);
        assert_eq!(r.altitude(), scalar * altitude);
    }
}

/// Tests the multiplication of a cylindrical vector by a scalar value with assignment.
#[test]
fn cylindrical_vec3d_multiplication_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let mut v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;

        // Perform multiplication by the scalar with assignment
        v *= scalar;

        // Verify components of the resulting vector
        assert_eq!(v.radius(), radius * scalar);
        assert_eq!(v.azimuth(), azimuth);
        assert_eq!(v.altitude(), altitude * scalar);
    }
}

/// Tests the division of a cylindrical vector by a scalar value.
#[test]
fn cylindrical_vec3d_divison_by_scalar_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        if scalar.abs() < shared::EPSILON {
            continue;
        }

        // Perform division by the scalar
        let r = v / scalar;

        // Verify components of the resulting vector
        assert_eq!(r.radius(), radius / scalar);
        assert_eq!(r.azimuth(), azimuth);
        assert_eq!(r.altitude(), altitude / scalar);
    }
}

/// Tests the division of a cylindrical vector by a scalar value with assignment.
#[test]
fn cylindrical_vec3d_divison_by_scalar_with_assignment_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..shared::ITERATIONS {
        // Generate random Cylindrical vector
        let (radius, azimuth, altitude) = shared::gen_cylindrical(&mut rng);
        let mut v = CylindricalBuilder::with(radius, azimuth, altitude).build();

        // Generate floating point scalar
        let scalar = (shared::MAX_VALUE - shared::MIN_VALUE) * rng.gen::<f64>() - shared::MIN_VALUE;
        if scalar.abs() < shared::EPSILON {
            continue;
        }

        // Perform division by the scalar with assignment
        v /= scalar;

        // Verify components of the resulting vector
        assert_eq!(v.radius(), radius / scalar);
        assert_eq!(v.azimuth(), azimuth);
        assert_eq!(v.altitude(), altitude / scalar);
    }
}
