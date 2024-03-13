#![allow(dead_code)]

use ephem::core::{matrices::*, vectors::CartesianBuilder};

mod shared;

#[test]
fn trivial_mat3d_creation_test() {
    let z = Mat3d::zeros();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(z[(i, j)], 0.0);
        }
    }

    let o = Mat3d::ones();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(o[(i, j)], 1.0);
        }
    }

    let e = Mat3d::eye();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(e[(i, j)], if i == j { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn mat3d_creation_with_rows_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let row1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let row2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let row3 = CartesianBuilder::with(x3, y3, z3).build();

        let m = Mat3d::with_rows(row1, row2, row3);

        assert_eq!(m[(0, 0)], x1);
        assert_eq!(m[(0, 1)], y1);
        assert_eq!(m[(0, 2)], z1);

        assert_eq!(m[(1, 0)], x2);
        assert_eq!(m[(1, 1)], y2);
        assert_eq!(m[(1, 2)], z2);

        assert_eq!(m[(2, 0)], x3);
        assert_eq!(m[(2, 1)], y3);
        assert_eq!(m[(2, 2)], z3);
    }
}

#[test]
fn mat3d_creation_with_columns_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let col1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let col2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let col3 = CartesianBuilder::with(x3, y3, z3).build();

        let m = Mat3d::with_columns(col1, col2, col3);

        assert_eq!(m[(0, 0)], x1);
        assert_eq!(m[(0, 1)], x2);
        assert_eq!(m[(0, 2)], x3);

        assert_eq!(m[(1, 0)], y1);
        assert_eq!(m[(1, 1)], y2);
        assert_eq!(m[(1, 2)], y3);

        assert_eq!(m[(2, 0)], z1);
        assert_eq!(m[(2, 1)], z2);
        assert_eq!(m[(2, 2)], z3);
    }
}
