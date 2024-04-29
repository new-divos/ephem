pub(crate) trait NegativelyProcessable<Lhs> {
    type Output;

    fn neg(lhs: &Lhs) -> Self::Output;
}

pub(crate) trait AdditivelyProcessable<Lhs, Rhs> {
    type Output;

    fn add(lhs: &Lhs, rhs: &Rhs) -> Self::Output;
    fn add_assign(lhs: &mut Lhs, rhs: &Rhs);

    fn sub(lhs: &Lhs, rhs: &Rhs) -> Self::Output;
    fn sub_assign(lhs: &mut Lhs, rhs: &Rhs);
}

pub(crate) trait MultiplyByScalarProcessable<Lhs> {
    type Output;

    fn mul(lhs: &Lhs, rhs: f64) -> Self::Output;
    fn mul_assign(lhs: &mut Lhs, rhs: f64);

    fn div(lhs: &Lhs, rhs: f64) -> Self::Output;
    fn div_assign(lhs: &mut Lhs, rhs: f64);
}
