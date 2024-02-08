#![allow(dead_code)]

use rand::Rng;

use ephem::core::vectors::*;

mod shared;

/// Test case for creating Cartesian vectors using the CartesianBuilder.
///
/// This test verifies the correctness of creating Cartesian vectors with various methods provided
/// by the CartesianBuilder struct. It checks that vectors are constructed correctly with zero
/// values, unit values, specific values, and random values within a range.
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
        let x = 200.0 * rng.gen::<f64>() - 100.0;
        let y = 200.0 * rng.gen::<f64>() - 100.0;
        let z = 200.0 * rng.gen::<f64>() - 100.0;

        // Test building a vector with specified values
        let v1 = CartesianBuilder::with(x, y, z).build();
        assert_eq!(v1.x(), x);
        assert_eq!(v1.y(), y);
        assert_eq!(v1.z(), z);

        // Test building a vector with method chaining
        let v2 = CartesianBuilder::new()
            .x(x)
            .y(y)
            .z(z)
            .build();
        assert_eq!(v2.x(), x);
        assert_eq!(v2.y(), y);
        assert_eq!(v2.z(), z);
    }
}