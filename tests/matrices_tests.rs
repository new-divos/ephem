#![allow(dead_code)]

use ephem::core::{matrices::*, vectors::CartesianBuilder};

mod shared;

/// Test function for trivial Mat3d creation methods.
#[test]
fn trivial_mat3d_creation_test() {
    // Test zeros matrix creation
    let z = Mat3d::zeros();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(z[(i, j)], 0.0);
        }
    }

    // Test ones matrix creation
    let o = Mat3d::ones();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(o[(i, j)], 1.0);
        }
    }

    // Test identity matrix creation
    let e = Mat3d::eye();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(e[(i, j)], if i == j { 1.0 } else { 0.0 });
        }
    }
}

/// Test function for Mat3d creation with rows.
#[test]
fn mat3d_creation_with_rows_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each row
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let row1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let row2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let row3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated rows
        let m = Mat3d::with_rows(row1, row2, row3);

        // Assert that each element of the resulting matrix corresponds to the correct Cartesian component
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

/// Test function for Mat3d creation with columns.
#[test]
fn mat3d_creation_with_columns_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each column
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let col1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let col2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let col3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated columns
        let m = Mat3d::with_columns(col1, col2, col3);

        // Assert that each element of the resulting matrix corresponds to the correct Cartesian component
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

/// Test function for indexing Mat3d rows through a view.
#[test]
fn mat3d_rows_view_index_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each row
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let row1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let row2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let row3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated rows
        let m = Mat3d::with_rows(row1, row2, row3);
        let rows = m.rows();

        // Assert that each element of the resulting view corresponds to the correct Cartesian component
        assert_eq!(rows[0].x(), x1);
        assert_eq!(rows[0].y(), y1);
        assert_eq!(rows[0].z(), z1);

        assert_eq!(rows[1].x(), x2);
        assert_eq!(rows[1].y(), y2);
        assert_eq!(rows[1].z(), z2);

        assert_eq!(rows[2].x(), x3);
        assert_eq!(rows[2].y(), y3);
        assert_eq!(rows[2].z(), z3);
    }
}

/// Test function for iterating over rows of a Mat3d.
#[test]
fn mat3d_rows_view_iterator_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each row
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let row1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let row2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let row3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated rows
        let m = Mat3d::with_rows(row1, row2, row3);
        let rows = m.rows();

        // Obtain an iterator from the view and iterate over the rows
        let mut rows_iter = rows.iter();

        let mut value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x1);
            assert_eq!(vector.y(), y1);
            assert_eq!(vector.z(), z1);
        }

        value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x2);
            assert_eq!(vector.y(), y2);
            assert_eq!(vector.z(), z2);
        }

        value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x3);
            assert_eq!(vector.y(), y3);
            assert_eq!(vector.z(), z3);
        }

        value = rows_iter.next();
        assert!(value.is_none());
    }
}

/// Test function for iterating over rows of a Mat3d using into_iter.
#[test]
fn mat3d_rows_view_into_iterator_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each row
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let row1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let row2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let row3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated rows
        let m = Mat3d::with_rows(row1, row2, row3);
        let rows = m.rows();

        // Obtain an iterator from the view and iterate over the rows
        let mut rows_iter = rows.into_iter();

        let mut value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x1);
            assert_eq!(vector.y(), y1);
            assert_eq!(vector.z(), z1);
        }

        value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x2);
            assert_eq!(vector.y(), y2);
            assert_eq!(vector.z(), z2);
        }

        value = rows_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x3);
            assert_eq!(vector.y(), y3);
            assert_eq!(vector.z(), z3);
        }

        value = rows_iter.next();
        assert!(value.is_none());
    }
}

/// Test function for accessing columns of a Mat3d using index.
#[test]
fn mat3d_columns_view_index_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each column
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let col1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let col2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let col3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated columns
        let m = Mat3d::with_columns(col1, col2, col3);
        let columns = m.columns();

        // Access the columns using index operations and assert the vector components
        assert_eq!(columns[0].x(), x1);
        assert_eq!(columns[0].y(), y1);
        assert_eq!(columns[0].z(), z1);

        assert_eq!(columns[1].x(), x2);
        assert_eq!(columns[1].y(), y2);
        assert_eq!(columns[1].z(), z2);

        assert_eq!(columns[2].x(), x3);
        assert_eq!(columns[2].y(), y3);
        assert_eq!(columns[2].z(), z3);
    }
}

/// Test function for iterating over columns of a Mat3d.
#[test]
fn mat3d_columns_view_iterator_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each column
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let col1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let col2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let col3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated columns
        let m = Mat3d::with_columns(col1, col2, col3);
        let columns = m.columns();

        // Create an iterator over the columns and assert the vector components
        let mut columns_iter = columns.iter();

        let mut value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x1);
            assert_eq!(vector.y(), y1);
            assert_eq!(vector.z(), z1);
        }

        value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x2);
            assert_eq!(vector.y(), y2);
            assert_eq!(vector.z(), z2);
        }

        value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x3);
            assert_eq!(vector.y(), y3);
            assert_eq!(vector.z(), z3);
        }

        value = columns_iter.next();
        assert!(value.is_none());
    }
}

/// Test function for iterating over columns of a Mat3d using into iterator.
#[test]
fn mat3d_columns_view_into_iterator_test() {
    let mut rng = rand::thread_rng();

    for _ in 0..shared::ITERATIONS {
        // Generate random Cartesian vectors for each column
        let (x1, y1, z1) = shared::gen_cartesian(&mut rng);
        let col1 = CartesianBuilder::with(x1, y1, z1).build();

        let (x2, y2, z2) = shared::gen_cartesian(&mut rng);
        let col2 = CartesianBuilder::with(x2, y2, z2).build();

        let (x3, y3, z3) = shared::gen_cartesian(&mut rng);
        let col3 = CartesianBuilder::with(x3, y3, z3).build();

        // Construct a Mat3d using the generated columns
        let m = Mat3d::with_columns(col1, col2, col3);
        let columns = m.columns();

        // Create an into iterator over the columns and assert the vector components
        let mut columns_iter = columns.into_iter();

        let mut value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x1);
            assert_eq!(vector.y(), y1);
            assert_eq!(vector.z(), z1);
        }

        value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x2);
            assert_eq!(vector.y(), y2);
            assert_eq!(vector.z(), z2);
        }

        value = columns_iter.next();
        assert!(value.is_some());
        if let Some(vector) = value {
            assert_eq!(vector.x(), x3);
            assert_eq!(vector.y(), y3);
            assert_eq!(vector.z(), z3);
        }

        value = columns_iter.next();
        assert!(value.is_none());
    }
}
