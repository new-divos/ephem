pub struct Mat3d([f64; 9]);

impl Mat3d {
    #[inline]
    pub fn zeros() -> Self {
        Self([0.0; 9])
    }

    #[inline]
    pub fn ones() -> Self {
        Self([1.0; 9])
    }

    #[inline]
    pub fn eye() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    pub fn r_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c])
    }

    pub fn r_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c])
    }

    pub fn f_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])
    }
}
