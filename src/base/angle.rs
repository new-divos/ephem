use std::cmp::Ordering;
use std::convert;
use std::f64::consts::PI;

use crate::base::consts::{ARCS, DEG, PI2, RAD};

const ARCM: f64 = 60.0 * 180.0 / PI;

const TMH: f64 = 12.0 / PI;
const TMM: f64 = 60.0 * 12.0 / PI;
const TMS: f64 = 3600.0 * 12.0 / PI;

const TMHTOARCM: f64 = 15.0 * 60.0;
const TMMTOARCS: f64 = 15.0 * 60.0;
const TMHTOARCS: f64 = TMHTOARCM * 60.0;
const ARCDTOTMS: f64 = 3600.0 / 15.0;

const RVARCD: f64 = 360.0;
const RVARCM: f64 = RVARCD * 60.0;
const RVARCS: f64 = RVARCM * 60.0;

const RVTMH: f64 = 24.0;
const RVTMM: f64 = RVTMH * 60.0;
const RVTMS: f64 = RVTMM * 60.0;


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Sign {
    Negative,
    Zero,
    Positive
}

trait AngleSign {
    fn sign(&self) -> Sign;
}

impl AngleSign for f64 {
    fn sign(&self) -> Sign {
        if *self == 0.0 {
            Sign::Zero
        } else if *self < 0.0 {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }
}


///
/// Left value in a composed angle value
///
#[derive(Debug, Copy, Clone, PartialEq)]
struct Left(f64);

///
/// Middle value in a composed angle value
///
#[derive(Debug, Copy, Clone, PartialEq)]
struct Middle(f64);

///
/// Right value in a composed angle value
///
#[derive(Debug, Copy, Clone, PartialEq)]
struct Right(f64);


#[derive(Debug, Copy, Clone, Default, PartialEq)]
struct ShortAngle(i32, f64);

impl AngleSign for ShortAngle {
    fn sign(&self) -> Sign {
        if self.0 == 0 && self.1 == 0.0 {
            Sign::Zero
        } else if self.0 < 0 || (self.0 == 0 && self.1 < 0.0) {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }
}

impl convert::Into<ShortAngle> for Left {
    fn into(self) -> ShortAngle {
        let v = self.0.abs();
        let u = v.floor();

        let result = ShortAngle(u as i32, 60.0 * (v - u));
        result.copysign(self.0)
    }
}

impl convert::Into<Left> for ShortAngle {
    fn into(self) -> Left {
        let mut value = (self.0.abs() as f64) + self.1.abs() / 60.0;
        if self.0 < 0 || (self.0 == 0 && self.1 < 0.0) {
            value = -value
        }

        Left(value)
    }
}

impl convert::Into<ShortAngle> for Right {
    fn into(self) -> ShortAngle {
        let v = self.0.abs();
        let u = (v / 60.0).floor();

        let result = ShortAngle(u as i32, v - 60.0 * u);
        result.copysign(self.0)
    }
}

impl convert::Into<Right> for ShortAngle {
    fn into(self) -> Right {
        if self.0 != 0 {
            let mut value = 60.0 * (self.0.abs() as f64) + self.1;
            if self.0 < 0 { value = -value }
            Right(value)
        } else {
            Right(self.1)
        }
    }
}

impl convert::Into<(i32, f64)> for ShortAngle {
    fn into(self) -> (i32, f64) {
        let ShortAngle(value1, value2) = self;
        (value1, value2)
    }
}

impl convert::Into<(Sign, i32, f64)> for ShortAngle {
    fn into(self) -> (Sign, i32, f64) {
        let ShortAngle(value1, value2) = self;
        (self.sign(), value1.abs(), value2.abs())
    }
}

impl PartialOrd for ShortAngle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.1.is_nan() || other.1.is_nan() {
            None
        } else {
            let Left(value1) = (*self).into();
            let Left(value2) = (*other).into();

            value1.partial_cmp(&value2)
        }
    }
}

impl ShortAngle {
    fn copysign(&self, value: f64) -> Self {
        let ShortAngle(value1, value2) = *self;

        let mut value1 = value1.abs();
        let mut value2 = value2.abs();

        if value < 0.0 {
            if value1 == 0 {
                value2 = -value2;
            } else {
                value1 = -value1;
            }
        }

        ShortAngle(value1, value2)
    }
}


#[derive(Debug, Copy, Clone, Default, PartialEq)]
struct LongAngle(i32, i8, f64);

impl AngleSign for LongAngle {
    fn sign(&self) -> Sign {
        if self.0 == 0 && self.1 == 0 && self.2 == 0.0 {
            Sign::Zero
        } else if self.0 < 0 || (
            self.0 == 0 && (self.1 < 0 || (self.1 == 0 && self.2 < 0.0))
        ) {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }
}

impl convert::Into<LongAngle> for Left {
    fn into(self) -> LongAngle {
        let v = self.0.abs();
        let u = v.floor();
        let w = 60.0 * (v - u);
        let m = w.floor();

        let result = LongAngle(u as i32, m as i8, 60.0 * (w - m));
        result.copysign(self.0)
    }
}

impl convert::Into<Left> for LongAngle {
    fn into(self) -> Left {
        let mut value = (self.0.abs() as f64) +
            ((self.1.abs() as f64) + self.2.abs() / 60.0) / 60.0;
        if self.0 < 0 || (self.0 == 0 &&
            (self.1 < 0 || (self.1 == 0 && self.2 < 0.0))
        ) { value = -value }

        Left(value)
    }
}

impl convert::Into<LongAngle> for Middle {
    fn into(self) -> LongAngle {
        let v = self.0.abs();
        let w = v.floor();
        let u = (w / 60.0).floor();
        let m = w - 60.0 * u;

        let result = LongAngle(u as i32, m as i8, 60.0 * (v - w));
        result.copysign(self.0)
    }
}

impl convert::Into<Middle> for LongAngle {
    fn into(self) -> Middle {
        if self.0 != 0 {
            let mut value =
                60.0 * (self.0.abs() as f64) + (self.1.abs() as f64) + self.2.abs() / 60.0;
            if self.0 < 0 { value = -value }
            Middle(value)
        } else if self.1 != 0 {
            let mut value = (self.1.abs() as f64) + self.2.abs() / 60.0;
            if self.1 < 0 { value = -value }
            Middle(value)
        } else {
            Middle(self.2 / 60.0)
        }
    }
}

impl convert::Into<LongAngle> for Right {
    fn into(self) -> LongAngle {
        let v = self.0.abs();
        let w = (v / 60.0).floor();
        let u = (w / 60.0).floor();
        let m = w - 60.0 * u;

        let result = LongAngle(u as i32, m as i8, v - 60.0 * w);
        result.copysign(self.0)
    }
}

impl convert::Into<Right> for LongAngle {
    fn into(self) -> Right {
        if self.0 != 0 {
            let mut value =
                self.2 + 60.0 * ((self.1 as f64) + 60.0 * (self.0.abs() as f64));
            if self.0 < 0 { value = -value }
            Right(value)
        } else if self.1 != 0 {
            let mut value = self.2 + 60.0 * (self.1.abs() as f64);
            if self.1 < 0 { value = -value }
            Right(value)
        } else {
            Right(self.2)
        }
    }
}

impl convert::Into<(i32, i32, f64)> for LongAngle {
    fn into(self) -> (i32, i32, f64) {
        let LongAngle(value1, value2, value3) = self;
        (value1, value2 as i32, value3)
    }
}

impl convert::Into<(Sign, i32, i32, f64)> for LongAngle {
    fn into(self) -> (Sign, i32, i32, f64) {
        let LongAngle(value1, value2, value3) = self;
        (self.sign(), value1.abs(), value2.abs() as i32, value3.abs())
    }
}

impl PartialOrd for LongAngle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.2.is_nan() || other.2.is_nan() {
            None
        } else {
            let Left(value1) = (*self).into();
            let Left(value2) = (*other).into();

            value1.partial_cmp(&value2)
        }
    }
}

impl LongAngle {
    fn copysign(&self, value: f64) -> Self {
        let LongAngle(value1, value2, value3) = *self;

        if value1 != 0 {
            let mut value1 = value1.abs();
            if value < 0.0 { value1 = -value1 };
            LongAngle(value1, value2, value3)
        } else if value2 != 0 {
            let mut value2 = value2.abs();
            if value < 0.0 { value2 = -value2 }
            LongAngle(value1, value2, value3)
        } else {
            LongAngle(value1, value2, value3.copysign(value))
        }
    }
}


macro_rules! impl_angle {
    ($t:ty; $value:ident) => {
        impl $t {
            #[inline]
            pub fn $value(&self) -> f64 {
                self.0.abs()
            }

            #[inline]
            pub fn sign(&self) -> Sign {
                self.0.sign()
            }

            #[inline]
            fn value(&self) -> f64 {
                self.0
            }
        }
    };
    ($t:ty; $value1:ident, $value2:ident) => {
        impl $t {
            pub fn $value1(&self) -> i32 {
                let Self(ShortAngle(value, _)) = *self;
                value.abs()
            }

            pub fn $value2(&self) -> f64 {
                let Self(ShortAngle(_, value)) = *self;
                value.abs()
            }

            #[inline]
            pub fn sigh(&self) -> Sign {
                self.0.sign()
            }

            fn value(&self) -> f64 {
                let Left(value) = self.0.into();
                value
            }

            fn new($value1: i32, $value2: f64) -> Self {
                let value: Right = ShortAngle($value1, $value2).into();
                Self(value.into())
            }
        }
    };
    ($t:ty; $value1:ident, $value2:ident, $value3:ident) => {
        impl $t {
            pub fn $value1(&self) -> i32 {
                let Self(LongAngle(value, ..)) = *self;
                value.abs()
            }

            pub fn $value2(&self) -> i32 {
                let Self(LongAngle(_, value, _)) = *self;
                (value as i32).abs()
            }

            pub fn $value3(&self) -> f64 {
                let Self(LongAngle(.., value)) = *self;
                value.abs()
            }

            #[inline]
            pub fn sign(&self) -> Sign {
                self.0.sign()
            }

            fn value(&self) -> f64 {
                let Left(value) = self.0.into();
                value
            }

            fn new($value1: i32, $value2: i32, $value3: f64) -> Self {
                let delta = $value2 / 60;
                let $value1 = $value1 + delta;
                let $value2 = ($value2 - 60 * delta) as i8;

                let value: Right = LongAngle($value1, $value2, $value3).into();
                Self(value.into())
            }
        }
    }
}


trait AngleOrd {}
trait AngleValue {}

trait UnpackAngleValue {}
trait UnpackShortAngle {}
trait RawShortAngle {}
trait UnpackLongAngle {}
trait RawLongAngle {}


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleRevolutions(f64);

impl_angle!(AngleRevolutions; revolutions);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleArcDegrees(f64);

impl_angle!(AngleArcDegrees; degrees);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawShortAngle, UnpackShortAngle)]
pub struct AngleArcDegreesMinutes(ShortAngle);

impl_angle!(AngleArcDegreesMinutes; degrees, minutes);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawLongAngle, UnpackLongAngle)]
pub struct AngleArcDegreesMinutesSeconds(LongAngle);

impl_angle!(AngleArcDegreesMinutesSeconds; degrees, minutes, seconds);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleArcMinutes(f64);

impl_angle!(AngleArcMinutes; minutes);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawShortAngle, UnpackShortAngle)]
pub struct AngleArcMinutesSeconds(ShortAngle);

impl_angle!(AngleArcMinutesSeconds; minutes, seconds);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleArcSeconds(f64);

impl_angle!(AngleArcSeconds; seconds);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleTimeHours(f64);

impl_angle!(AngleTimeHours; hours);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawShortAngle, UnpackShortAngle)]
pub struct AngleTimeHoursMinutes(ShortAngle);

impl_angle!(AngleTimeHoursMinutes; hours, minutes);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawLongAngle, UnpackLongAngle)]
pub struct AngleTimeHoursMinutesSeconds(LongAngle);

impl_angle!(AngleTimeHoursMinutesSeconds; hours, minutes, seconds);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleTimeMinutes(f64);

impl_angle!(AngleTimeMinutes; minutes);


#[derive(Debug, Copy, Clone, Default, PartialEq,
    AngleOrd, AngleValue, RawShortAngle, UnpackShortAngle)]
pub struct AngleTimeMinutesSeconds(ShortAngle);

impl_angle!(AngleTimeMinutesSeconds; minutes, seconds);


#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd,
    AngleValue, UnpackAngleValue)]
pub struct AngleTimeSeconds(f64);

impl_angle!(AngleTimeSeconds; seconds);


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Angle {
    Radians(f64),
    Revolutions(AngleRevolutions),
    ArcDegrees(AngleArcDegrees),
    ArcDegreesMinutes(AngleArcDegreesMinutes),
    ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds),
    ArcMinutes(AngleArcMinutes),
    ArcMinutesSeconds(AngleArcMinutesSeconds),
    ArcSeconds(AngleArcSeconds),
    TimeHours(AngleTimeHours),
    TimeHoursMinutes(AngleTimeHoursMinutes),
    TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds),
    TimeMinutes(AngleTimeMinutes),
    TimeMinutesSeconds(AngleTimeMinutesSeconds),
    TimeSeconds(AngleTimeSeconds)
}

impl Default for Angle {
    fn default() -> Self {
        Angle::Radians(0.0)
    }
}

impl convert::From<f64> for Angle {
    fn from(radians: f64) -> Self {
        Angle::Radians(radians)
    }
}


impl convert::Into<f64> for Angle {
    fn into(self) -> f64 {
        match self {
            Angle::Radians(r) => r,
            Angle::Revolutions(AngleRevolutions(r)) => {
                r * PI2
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                d * RAD
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                degrees * RAD
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                degrees * RAD
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                m / ARCM
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                minutes / ARCM
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                s / ARCS
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                h / TMH
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                hours / TMH
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                hours / TMH
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                m / TMM
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                minutes / TMM
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                s / TMS
            }
        }
    }
}

impl convert::Into<AngleRevolutions> for Angle {
    fn into(self) -> AngleRevolutions {
        match self {
            Angle::Radians(r) => AngleRevolutions(r / PI2),
            Angle::Revolutions(r) => r,
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleRevolutions(d / RVARCD)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleRevolutions(degrees / RVARCD)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleRevolutions(degrees / RVARCD)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleRevolutions(m / RVARCM)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleRevolutions(minutes / RVARCM)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleRevolutions(s / RVARCS)
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleRevolutions(h / RVTMH)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                AngleRevolutions(hours / RVTMH)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                AngleRevolutions(hours / RVTMH)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleRevolutions(m / RVTMM)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleRevolutions(minutes / RVTMM)
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleRevolutions(s / RVTMS)
            }
        }
    }
}

impl convert::Into<Option<AngleRevolutions>> for Angle {
    fn into(self) -> Option<AngleRevolutions> {
        match self {
            Angle::Revolutions(r) => Some(r),
            _ => None
        }
    }
}

impl convert::Into<AngleArcDegrees> for Angle {
    fn into(self) -> AngleArcDegrees {
        match self {
            Angle::Radians(r) => {
                AngleArcDegrees(DEG * r)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcDegrees(RVARCD * r)
            },
            Angle::ArcDegrees(d) => d,
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleArcDegrees(degrees)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleArcDegrees(degrees)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleArcDegrees(m / 60.0)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegrees(minutes / 60.0)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleArcDegrees(s / 3600.0)
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcDegrees(h * 15.0)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                AngleArcDegrees(hours * 15.0)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                AngleArcDegrees(hours * 15.0)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcDegrees(m / 4.0)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegrees(minutes / 4.0)
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcDegrees(s / 240.0)
            }
        }
    }
}

impl convert::Into<Option<AngleArcDegrees>> for Angle {
    fn into(self) -> Option<AngleArcDegrees> {
        match self {
            Angle::ArcDegrees(d) => Some(d),
            _ => None
        }
    }
}

impl convert::Into<AngleArcDegreesMinutes> for Angle {
    fn into(self) -> AngleArcDegreesMinutes {
        match self {
            Angle::Radians(r) => {
                AngleArcDegreesMinutes(Left(DEG * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcDegreesMinutes(Left(RVARCD * r).into())
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleArcDegreesMinutes(Left(d).into())
            },
            Angle::ArcDegreesMinutes(dm) => dm,
            Angle::ArcDegreesMinutesSeconds(
                AngleArcDegreesMinutesSeconds(LongAngle(d, m, s))
            ) => {
                let ms = ShortAngle(m as i32, s);
                let Left(minutes) = ms.into();
                AngleArcDegreesMinutes(ShortAngle(d, minutes))
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleArcDegreesMinutes(Right(m).into())
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegreesMinutes(Right(minutes).into())
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleArcDegreesMinutes(Right(s / 60.0).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcDegreesMinutes(Left(15.0 * h).into())
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                AngleArcDegreesMinutes(Left(15.0 * hours).into())
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                AngleArcDegreesMinutes(Left(15.0 * hours).into())
            }
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcDegreesMinutes(Right(15.0 * m).into())
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegreesMinutes(Right(15.0 * minutes).into())
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcDegreesMinutes(Right(s / 4.0).into())
            }
        }
    }
}

impl convert::Into<Option<AngleArcDegreesMinutes>> for Angle {
    fn into(self) -> Option<AngleArcDegreesMinutes> {
        match self {
            Angle::ArcDegreesMinutes(dm) => Some(dm),
            _ => None
        }
    }
}

impl convert::Into<AngleArcDegreesMinutesSeconds> for Angle {
    fn into(self) -> AngleArcDegreesMinutesSeconds {
        match self {
            Angle::Radians(r) => {
                AngleArcDegreesMinutesSeconds(Left(DEG * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcDegreesMinutesSeconds(Left(RVARCD * r).into())
            }
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleArcDegreesMinutesSeconds(Left(d).into())
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(ShortAngle(d, m))) => {
                let ShortAngle(m, s) = Left(m).into();
                AngleArcDegreesMinutesSeconds(LongAngle(d, m as i8, s))
            },
            Angle::ArcDegreesMinutesSeconds(dms) => dms,
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleArcDegreesMinutesSeconds(Middle(m).into())
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegreesMinutesSeconds(Middle(minutes).into())
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleArcDegreesMinutesSeconds(Right(s).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcDegreesMinutesSeconds(Left(15.0 * h).into())
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                AngleArcDegreesMinutesSeconds(Left(15.0 * hours).into())
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                AngleArcDegreesMinutesSeconds(Left(15.0 * hours).into())
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcDegreesMinutesSeconds(Middle(15.0 * m).into())
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcDegreesMinutesSeconds(Middle(15.0 * minutes).into())
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcDegreesMinutesSeconds(Right(15.0 * s).into())
            }
        }
    }
}

impl convert::Into<Option<AngleArcDegreesMinutesSeconds>> for Angle {
    fn into(self) -> Option<AngleArcDegreesMinutesSeconds> {
        match self {
            Angle::ArcDegreesMinutesSeconds(dms) => Some(dms),
            _ => None
        }
    }
}

impl convert::Into<AngleArcMinutes> for Angle {
    fn into(self) -> AngleArcMinutes {
        match self {
            Angle::Radians(r) => {
                AngleArcMinutes(ARCM * r)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcMinutes(RVARCM * r)
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleArcMinutes(60.0 * d)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Right(minutes) = dm.into();
                AngleArcMinutes(minutes)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Middle(minutes) = dms.into();
                AngleArcMinutes(minutes)
            },
            Angle::ArcMinutes(m) => m,
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcMinutes(minutes)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleArcMinutes(s / 60.0)
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcMinutes(TMHTOARCM * h)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleArcMinutes(15.0 * minutes)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Middle(minutes) = hms.into();
                AngleArcMinutes(15.0 * minutes)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcMinutes(15.0 * m)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcMinutes(15.0 * minutes)
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcMinutes(s / 4.0)
            }
        }
    }
}

impl convert::Into<Option<AngleArcMinutes>> for Angle {
    fn into(self) -> Option<AngleArcMinutes> {
        match self {
            Angle::ArcMinutes(m) => Some(m),
            _ => None
        }
    }
}

impl convert::Into<AngleArcMinutesSeconds> for Angle {
    fn into(self) -> AngleArcMinutesSeconds {
        match self {
            Angle::Radians(r) => {
                AngleArcMinutesSeconds(Left(ARCM * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcMinutesSeconds(Left(RVARCM * r).into())
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleArcMinutesSeconds(Left(60.0 * d).into())
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Right(minutes) = dm.into();
                AngleArcMinutesSeconds(Left(minutes).into())
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Middle(minutes) = dms.into();
                AngleArcMinutesSeconds(Left(minutes).into())
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleArcMinutesSeconds(Left(m).into())
            },
            Angle::ArcMinutesSeconds(ms) => ms,
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleArcMinutesSeconds(Right(s).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcMinutesSeconds(Left(TMHTOARCM * h).into())
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleArcMinutesSeconds(Left(15.0 * minutes).into())
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Middle(minutes) = hms.into();
                AngleArcMinutesSeconds(Left(15.0 * minutes).into())
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcMinutesSeconds(Left(15.0 * m).into())
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleArcMinutesSeconds(Left(15.0 * minutes).into())
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcMinutesSeconds(Right(15.0 * s).into())
            }
        }
    }
}

impl convert::Into<Option<AngleArcMinutesSeconds>> for Angle {
    fn into(self) -> Option<AngleArcMinutesSeconds> {
        match self {
            Angle::ArcMinutesSeconds(ms) => Some(ms),
            _ => None
        }
    }
}

impl convert::Into<AngleArcSeconds> for Angle {
    fn into(self) -> AngleArcSeconds {
        match self {
            Angle::Radians(r) => {
                AngleArcSeconds(ARCS * r)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleArcSeconds(RVARCS * r)
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleArcSeconds(3600.0 * d)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Right(minutes) = dm.into();
                AngleArcSeconds(60.0 * minutes)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Right(seconds) = dms.into();
                AngleArcSeconds(seconds)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleArcSeconds(60.0 * m)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleArcSeconds(seconds)
            },
            Angle::ArcSeconds(s) => s,
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleArcSeconds(TMHTOARCS * h)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleArcSeconds(TMMTOARCS * minutes)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Right(seconds) = hms.into();
                AngleArcSeconds(15.0 * seconds)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleArcSeconds(TMMTOARCS * m)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleArcSeconds(15.0 * seconds)
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleArcSeconds(15.0 * s)
            }
        }
    }
}

impl convert::Into<Option<AngleArcSeconds>> for Angle {
    fn into(self) -> Option<AngleArcSeconds> {
        match self {
            Angle::ArcSeconds(s) => Some(s),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeHours> for Angle {
    fn into(self) -> AngleTimeHours {
        match self {
            Angle::Radians(r) => {
                AngleTimeHours(TMH * r)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeHours(RVTMH * r)
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeHours(d / 15.0)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleTimeHours(degrees / 15.0)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleTimeHours(degrees / 15.0)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeHours(m / TMHTOARCM)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeHours(minutes / TMHTOARCM)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeHours(s / TMHTOARCS)
            },
            Angle::TimeHours(h) => h,
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Left(hours) = hm.into();
                AngleTimeHours(hours)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Left(hours) = hms.into();
                AngleTimeHours(hours)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleTimeHours(m / 60.0)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeHours(minutes / 60.0)
            }
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleTimeHours(s / 3600.0)
            }
        }
    }
}

impl convert::Into<Option<AngleTimeHours>> for Angle {
    fn into(self) -> Option<AngleTimeHours> {
        match self {
            Angle::TimeHours(h) => Some(h),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeHoursMinutes> for Angle {
    fn into(self) -> AngleTimeHoursMinutes {
        match self {
            Angle::Radians(r) => {
                AngleTimeHoursMinutes(Left(TMH * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeHoursMinutes(Left(RVTMH * r).into())
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeHoursMinutes(Left(d / 15.0).into())
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleTimeHoursMinutes(Left(degrees / 15.0).into())
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds (dms)) => {
                let Left(degrees) = dms.into();
                AngleTimeHoursMinutes(Left(degrees / 15.0).into())
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeHoursMinutes(Right(m / 15.0).into())
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeHoursMinutes(Right(minutes / 15.0).into())
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeHoursMinutes(Right(s / TMMTOARCS).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleTimeHoursMinutes(Left(h).into())
            },
            Angle::TimeHoursMinutes(hm) => hm,
            Angle::TimeHoursMinutesSeconds(
                AngleTimeHoursMinutesSeconds(LongAngle(h, m, s))
            ) => {
                let short = ShortAngle(m as i32, s);
                let Left(minutes) = short.into();
                AngleTimeHoursMinutes(ShortAngle(h, minutes))
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleTimeHoursMinutes(Right(m).into())
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeHoursMinutes(Right(minutes).into())
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleTimeHoursMinutes(Right(s / 60.0).into())
            }
        }
    }
}

impl convert::Into<Option<AngleTimeHoursMinutes>> for Angle {
    fn into(self) -> Option<AngleTimeHoursMinutes> {
        match self {
            Angle::TimeHoursMinutes(tm) => Some(tm),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeHoursMinutesSeconds> for Angle {
    fn into(self) -> AngleTimeHoursMinutesSeconds {
        match self {
            Angle::Radians(r) => {
                AngleTimeHoursMinutesSeconds(Left(TMH * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeHoursMinutesSeconds(Left(RVTMH * r).into())
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeHoursMinutesSeconds(Left(d / 15.0).into())
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleTimeHoursMinutesSeconds(Left(degrees / 15.0).into())
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleTimeHoursMinutesSeconds(Left(degrees / 15.0).into())
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeHoursMinutesSeconds(Middle(m / 15.0).into())
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleTimeHoursMinutesSeconds(Right(seconds / 15.0).into())
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeHoursMinutesSeconds(Right(s / 15.0).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleTimeHoursMinutesSeconds(Left(h).into())
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(ShortAngle(h, m))) => {
                let ShortAngle(m, s) = Left(m).into();
                AngleTimeHoursMinutesSeconds(LongAngle(h, m as i8, s))
            },
            Angle::TimeHoursMinutesSeconds(dms) => dms,
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleTimeHoursMinutesSeconds(Middle(m).into())
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleTimeHoursMinutesSeconds(Right(seconds).into())
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleTimeHoursMinutesSeconds(Right(s).into())
            }
        }
    }
}

impl convert::Into<Option<AngleTimeHoursMinutesSeconds>> for Angle {
    fn into(self) -> Option<AngleTimeHoursMinutesSeconds> {
        match self {
            Angle::TimeHoursMinutesSeconds(hms) => Some(hms),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeMinutes> for Angle {
    fn into(self) -> AngleTimeMinutes {
        match self {
            Angle::Radians(r) => {
                AngleTimeMinutes(TMM * r)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeMinutes(RVTMM * r)
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeMinutes(4.0 * d)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleTimeMinutes(4.0 * degrees)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleTimeMinutes(4.0 * degrees)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeMinutes(m / 15.0)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeMinutes(minutes / 15.0)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeMinutes(s / TMMTOARCS)
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleTimeMinutes(60.0 * h)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleTimeMinutes(minutes)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Middle(minutes) = hms.into();
                AngleTimeMinutes(minutes)
            },
            Angle::TimeMinutes(m) => m,
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeMinutes(minutes)
            },
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleTimeMinutes(s / 60.0)
            }
        }
    }
}

impl convert::Into<Option<AngleTimeMinutes>> for Angle {
    fn into(self) -> Option<AngleTimeMinutes> {
        match self {
            Angle::TimeMinutes(m) => Some(m),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeMinutesSeconds> for Angle {
    fn into(self) -> AngleTimeMinutesSeconds {
        match self {
            Angle::Radians(r) => {
                AngleTimeMinutesSeconds(Left(TMM * r).into())
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeMinutesSeconds(Left(RVTMM * r).into())
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeMinutesSeconds(Left(4.0 * d).into())
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Left(degrees) = dm.into();
                AngleTimeMinutesSeconds(Left(4.0 * degrees).into())
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Left(degrees) = dms.into();
                AngleTimeMinutesSeconds(Left(4.0 * degrees).into())
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeMinutesSeconds(Left(m / 15.0).into())
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Left(minutes) = ms.into();
                AngleTimeMinutesSeconds(Left(minutes / 15.0).into())
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeMinutesSeconds(Left(s / TMMTOARCS).into())
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleTimeMinutesSeconds(Left(60.0 * h).into())
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleTimeMinutesSeconds(Left(minutes).into())
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Middle(minutes) = hms.into();
                AngleTimeMinutesSeconds(Left(minutes).into())
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleTimeMinutesSeconds(Left(m).into())
            },
            Angle::TimeMinutesSeconds(ms) => ms,
            Angle::TimeSeconds(AngleTimeSeconds(s)) => {
                AngleTimeMinutesSeconds(Right(s).into())
            }
        }
    }
}

impl convert::Into<Option<AngleTimeMinutesSeconds>> for Angle {
    fn into(self) -> Option<AngleTimeMinutesSeconds> {
        match self {
            Angle::TimeMinutesSeconds(tm) => Some(tm),
            _ => None
        }
    }
}

impl convert::Into<AngleTimeSeconds> for Angle {
    fn into(self) -> AngleTimeSeconds {
        match self {
            Angle::Radians(r) => {
                AngleTimeSeconds(r * TMS)
            },
            Angle::Revolutions(AngleRevolutions(r)) => {
                AngleTimeSeconds(r * RVTMS)
            },
            Angle::ArcDegrees(AngleArcDegrees(d)) => {
                AngleTimeSeconds(d * ARCDTOTMS)
            },
            Angle::ArcDegreesMinutes(AngleArcDegreesMinutes(dm)) => {
                let Right(minutes) = dm.into();
                AngleTimeSeconds(4.0 * minutes)
            },
            Angle::ArcDegreesMinutesSeconds(AngleArcDegreesMinutesSeconds(dms)) => {
                let Right(seconds) = dms.into();
                AngleTimeSeconds(seconds / 15.0)
            },
            Angle::ArcMinutes(AngleArcMinutes(m)) => {
                AngleTimeSeconds(4.0 * m)
            },
            Angle::ArcMinutesSeconds(AngleArcMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleTimeSeconds(seconds / 15.0)
            },
            Angle::ArcSeconds(AngleArcSeconds(s)) => {
                AngleTimeSeconds(s / 15.0)
            },
            Angle::TimeHours(AngleTimeHours(h)) => {
                AngleTimeSeconds(h * 3600.0)
            },
            Angle::TimeHoursMinutes(AngleTimeHoursMinutes(hm)) => {
                let Right(minutes) = hm.into();
                AngleTimeSeconds(minutes * 60.0)
            },
            Angle::TimeHoursMinutesSeconds(AngleTimeHoursMinutesSeconds(hms)) => {
                let Right(seconds) = hms.into();
                AngleTimeSeconds(seconds)
            },
            Angle::TimeMinutes(AngleTimeMinutes(m)) => {
                AngleTimeSeconds(m * 60.0)
            },
            Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds(ms)) => {
                let Right(seconds) = ms.into();
                AngleTimeSeconds(seconds)
            },
            Angle::TimeSeconds(s) => s
        }
    }
}

impl convert::Into<Option<AngleTimeSeconds>> for Angle {
    fn into(self) -> Option<AngleTimeSeconds> {
        match self {
            Angle::TimeSeconds(s) => Some(s),
            _ => None
        }
    }
}

impl PartialOrd for Angle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else {
            let value1: f64 = (*self).into();
            let value2: f64 = (*other).into();

            value1.partial_cmp(&value2)
        }
    }
}

impl Angle {
    pub fn from_r(revolutions: f64) -> Angle {
        Angle::Revolutions(AngleRevolutions(revolutions))
    }

    pub fn from_ad(degrees: f64) -> Angle {
        Angle::ArcDegrees(AngleArcDegrees(degrees))
    }

    pub fn from_adm(degrees: i32, minutes: f64) -> Angle {
        Angle::ArcDegreesMinutes(AngleArcDegreesMinutes::new(degrees, minutes))
    }

    pub fn from_adms(degrees: i32, minutes: i32, seconds: f64) -> Angle {
        Angle::ArcDegreesMinutesSeconds(
            AngleArcDegreesMinutesSeconds::new(degrees, minutes, seconds)
        )
    }

    pub fn from_am(minutes: f64) -> Angle {
        Angle::ArcMinutes(AngleArcMinutes(minutes))
    }

    pub fn from_ams(minutes: i32, seconds: f64) -> Angle {
        Angle::ArcMinutesSeconds(AngleArcMinutesSeconds::new(minutes, seconds))
    }

    pub fn from_as(seconds: f64) -> Angle {
        Angle::ArcSeconds(AngleArcSeconds(seconds))
    }

    pub fn from_th(hours: f64) -> Angle {
        Angle::TimeHours(AngleTimeHours(hours))
    }

    pub fn from_thm(hours: i32, minutes: f64) -> Angle {
        Angle::TimeHoursMinutes(AngleTimeHoursMinutes::new(hours, minutes))
    }

    pub fn from_thms(hours: i32, minutes: i32, seconds: f64) -> Angle {
        Angle::TimeHoursMinutesSeconds(
            AngleTimeHoursMinutesSeconds::new(hours, minutes, seconds)
        )
    }

    pub fn from_tm(minutes: f64) -> Angle {
        Angle::TimeMinutes(AngleTimeMinutes(minutes))
    }

    pub fn from_tms(minutes: i32, seconds: f64) -> Angle {
        Angle::TimeMinutesSeconds(AngleTimeMinutesSeconds::new(minutes, seconds))
    }

    pub fn from_ts(seconds: f64) -> Angle {
        Angle::TimeSeconds(AngleTimeSeconds(seconds))
    }

    pub fn is_rad(&self) -> bool {
        match self {
            Angle::Radians(_) => true,
            _ => false
        }
    }

    pub fn is_r(&self) -> bool {
        match self {
            Angle::Revolutions(_) => true,
            _ => false
        }
    }

    pub fn is_ad(&self) -> bool {
        match self {
            Angle::ArcDegrees(_) => true,
            _ => false
        }
    }

    pub fn is_adm(&self) -> bool {
        match self {
            Angle::ArcDegreesMinutes(_) => true,
            _ => false
        }
    }

    pub fn is_adms(&self) -> bool {
        match self {
            Angle::ArcDegreesMinutesSeconds(_) => true,
            _ => false
        }
    }

    pub fn is_am(&self) -> bool {
        match self {
            Angle::ArcMinutes(_) => true,
            _ => false
        }
    }

    pub fn is_ams(&self) -> bool {
        match self {
            Angle::ArcMinutesSeconds(_) => true,
            _ => false
        }
    }

    pub fn is_as(&self) -> bool {
        match self {
            Angle::ArcSeconds(_) => true,
            _ => false
        }
    }

    pub fn is_th(&self) -> bool {
        match self {
            Angle::TimeHours(_) => true,
            _ => false
        }
    }

    pub fn is_thm(&self) -> bool {
        match self {
            Angle::TimeHoursMinutes(_) => true,
            _ => false
        }
    }

    pub fn is_thms(&self) -> bool {
        match self {
            Angle::TimeHoursMinutesSeconds(_) => true,
            _ => false
        }
    }

    pub fn is_tm(&self) -> bool {
        match self {
            Angle::TimeMinutes(_) => true,
            _ => false
        }
    }

    pub fn is_tms(&self) -> bool {
        match self {
            Angle::TimeMinutesSeconds(_) => true,
            _ => false
        }
    }

    pub fn is_ts(&self) -> bool {
        match self {
            Angle::TimeSeconds(_) => true,
            _ => false
        }
    }

    pub fn to_rad(self) -> Angle {
        Angle::Radians(self.into())
    }

    pub fn to_r(self) -> Angle {
        Angle::Revolutions(self.into())
    }

    pub fn to_ad(self) -> Angle {
        Angle::ArcDegrees(self.into())
    }

    pub fn to_adm(self) -> Angle {
        Angle::ArcDegreesMinutes(self.into())
    }

    pub fn to_adms(self) -> Angle {
        Angle::ArcDegreesMinutesSeconds(self.into())
    }

    pub fn to_am(self) -> Angle {
        Angle::ArcMinutes(self.into())
    }

    pub fn to_ams(self) -> Angle {
        Angle::ArcMinutesSeconds(self.into())
    }

    pub fn to_as(self) -> Angle {
        Angle::ArcSeconds(self.into())
    }

    pub fn to_th(self) -> Angle {
        Angle::TimeHours(self.into())
    }

    pub fn to_thm(self) -> Angle {
        Angle::TimeHoursMinutes(self.into())
    }

    pub fn to_thms(self) -> Angle {
        Angle::TimeHoursMinutesSeconds(self.into())
    }

    pub fn to_tm(self) -> Angle {
        Angle::TimeMinutes(self.into())
    }

    pub fn to_tms(self) -> Angle {
        Angle::TimeMinutesSeconds(self.into())
    }

    pub fn to_ts(self) -> Angle {
        Angle::TimeSeconds(self.into())
    }

    fn is_nan(&self) -> bool {
        match *self {
            Angle::Radians(r) => r.is_nan(),
            Angle::Revolutions(AngleRevolutions(r)) => r.is_nan(),
            Angle::ArcDegrees(AngleArcDegrees(d)) => d.is_nan(),
            Angle::ArcDegreesMinutes(
                AngleArcDegreesMinutes(ShortAngle(_, m))
            ) => m.is_nan(),
            Angle::ArcDegreesMinutesSeconds(
                AngleArcDegreesMinutesSeconds(LongAngle(_, _, s))
            ) => s.is_nan(),
            Angle::ArcMinutes(AngleArcMinutes(m)) => m.is_nan(),
            Angle::ArcMinutesSeconds(
                AngleArcMinutesSeconds(ShortAngle(_, s))
            ) => s.is_nan(),
            Angle::ArcSeconds(AngleArcSeconds(s)) => s.is_nan(),
            Angle::TimeHours(AngleTimeHours(h)) => h.is_nan(),
            Angle::TimeHoursMinutes(
                AngleTimeHoursMinutes(ShortAngle(_, m))
            ) => m.is_nan(),
            Angle::TimeHoursMinutesSeconds(
                AngleTimeHoursMinutesSeconds(LongAngle(_, _, s))
            ) => s.is_nan(),
            Angle::TimeMinutes(AngleTimeMinutes(m)) => m.is_nan(),
            Angle::TimeMinutesSeconds(
                AngleTimeMinutesSeconds(ShortAngle(_, s))
            ) => s.is_nan(),
            Angle::TimeSeconds(AngleTimeSeconds(s)) => s.is_nan()
        }
    }
}


#[cfg(test)]
mod tests {
    use rand::{Rng, thread_rng};
    use rand::distributions::Uniform;

    use super::*;

    const EPS: f64 = 1e-8;
    const ITERATIONS: i32 = 200;

    #[test]
    fn short_angle_test() {
        let short = ShortAngle(30, 30.0);
        let Left(t) = short.into();
        assert_relative_eq!(t, 30.5);
        let Right(t) = short.into();
        assert_relative_eq!(t, 30.0 * 60.0 + 30.0);

        let short: ShortAngle = Left(30.5).into();
        assert_eq!(short.0, 30);
        assert_relative_eq!(short.1, 30.0);
        let short: ShortAngle = Right(30.0 * 60.0 + 30.0).into();
        assert_eq!(short.0, 30);
        assert_relative_eq!(short.1, 30.0);

        let short = ShortAngle(-30, 36.0);
        let Left(t) = short.into();
        assert_relative_eq!(t, -30.6);
        let Right(t) = short.into();
        assert_relative_eq!(t, -(30.0 * 60.0 + 36.0));

        let short: ShortAngle = Left(-30.6).into();
        assert_eq!(short.0, -30);
        assert_relative_eq!(short.1, 36.0, epsilon = EPS);
        let short: ShortAngle = Right(-(30.0 * 60.0 + 36.0)).into();
        assert_eq!(short.0, -30);
        assert_relative_eq!(short.1, 36.0, epsilon = EPS);

        let short = ShortAngle(0, 36.0);
        let Left(t) = short.into();
        assert_relative_eq!(t, 0.6);
        let Right(t) = short.into();
        assert_relative_eq!(t, 36.0);

        let short: ShortAngle = Left(0.6).into();
        assert_eq!(short.0, 0);
        assert_relative_eq!(short.1, 36.0, epsilon = EPS);
        let short: ShortAngle = Right(36.0).into();
        assert_eq!(short.0, 0);
        assert_relative_eq!(short.1, 36.0, epsilon = EPS);

        let short = ShortAngle(0, -36.0);
        let Left(t) = short.into();
        assert_relative_eq!(t, -0.6);
        let Right(t) = short.into();
        assert_relative_eq!(t, -36.0);

        let short: ShortAngle = Left(-0.6).into();
        assert_eq!(short.0, 0);
        assert_relative_eq!(short.1, -36.0, epsilon = EPS);
        let short: ShortAngle = Right(-36.0).into();
        assert_eq!(short.0, 0);
        assert_relative_eq!(short.1, -36.0, epsilon = EPS);

        let mut rng = thread_rng();
        let side = Uniform::new(-360.0_f64, 360.0_f64);

        let left = Uniform::new(-360i32, 360i32);
        let right = Uniform::new(0.0_f64, 60.0_f64);

        for _ in 0..ITERATIONS {
            let left_value = rng.sample(side);

            let short: ShortAngle = Left(left_value).into();
            let Left(t) = short.into();
            assert_relative_eq!(t, left_value);

            let value1 = rng.sample(left);
            let value2 = rng.sample(right);

            let short = ShortAngle(value1, value2);
            let value: Left = short.into();
            let ShortAngle(t1, t2) = value.into();
            assert_eq!(t1, value1);
            assert_relative_eq!(t2, value2, epsilon = EPS);

            let right_value = 60.0 * left_value;

            let short: ShortAngle = Right(right_value).into();
            let Right(t) = short.into();
            assert_relative_eq!(t, right_value);
            let Left(t) = short.into();
            assert_relative_eq!(t, left_value);

            let short = ShortAngle(value1, value2);
            let value: Right = short.into();
            let ShortAngle(t1, t2) = value.into();
            assert_eq!(t1, value1);
            assert_relative_eq!(t2, value2, epsilon = EPS);
        }
    }

    #[test]
    fn long_angle_test() {
        let long = LongAngle(16, 14, 4.2);
        let Left(t) = long.into();
        assert_relative_eq!(t, 16.2345);
        let Middle(t) = long.into();
        assert_relative_eq!(t, 60.0 * 16.0 + 14.07);
        let Right(t) = long.into();
        assert_relative_eq!(t, 4.2 + 60.0 * (14.0 + 60.0 * 16.0));

        let long: LongAngle = Left(16.2345).into();
        assert_eq!(long.0, 16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Middle(60.0 * 16.0 + 14.07).into();
        assert_eq!(long.0, 16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Right(4.2 + 60.0 * (14.0 + 60.0 * 16.0)).into();
        assert_eq!(long.0, 16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long = LongAngle(-16, 14, 4.2);
        let Left(t) = long.into();
        assert_relative_eq!(t, -16.2345);
        let Middle(t) = long.into();
        assert_relative_eq!(t, -(60.0 * 16.0 + 14.07));
        let Right(t) = long.into();
        assert_relative_eq!(t, -(4.2 + 60.0 * (14.0 + 60.0 * 16.0)));

        let long: LongAngle = Left(-16.2345).into();
        assert_eq!(long.0, -16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Middle(-(60.0 * 16.0 + 14.07)).into();
        assert_eq!(long.0, -16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Right(-(4.2 + 60.0 * (14.0 + 60.0 * 16.0))).into();
        assert_eq!(long.0, -16);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long = LongAngle(0, 14, 4.2);
        let Left(t) = long.into();
        assert_relative_eq!(t, 0.2345);
        let Middle(t) = long.into();
        assert_relative_eq!(t, 14.07);
        let Right(t) = long.into();
        assert_relative_eq!(t, 4.2 + 60.0 * 14.0);

        let long: LongAngle = Left(0.2345).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Middle(14.07).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Right(4.2 + 60.0 * 14.0).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long = LongAngle(0, -14, 4.2);
        let Left(t) = long.into();
        assert_relative_eq!(t, -0.2345);
        let Middle(t) = long.into();
        assert_relative_eq!(t, -14.07);
        let Right(t) = long.into();
        assert_relative_eq!(t, -(4.2 + 60.0 * 14.0));

        let long: LongAngle = Left(-0.2345).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, -14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Middle(-14.07).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, -14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long: LongAngle = Right(-(4.2 + 60.0 * 14.0)).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, -14);
        assert_relative_eq!(long.2, 4.2, epsilon = EPS);

        let long = LongAngle(0, 0, 36.0);
        let Left(t) = long.into();
        assert_relative_eq!(t, 0.01);
        let Middle(t) = long.into();
        assert_relative_eq!(t, 0.6);
        let Right(t) = long.into();
        assert_relative_eq!(t, 36.0);

        let long: LongAngle = Left(0.01).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, 36.0, epsilon = EPS);

        let long: LongAngle = Middle(0.6).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, 36.0, epsilon = EPS);

        let long: LongAngle = Right(36.0).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, 36.0, epsilon = EPS);

        let long = LongAngle(0, 0, -36.0);
        let Left(t) = long.into();
        assert_relative_eq!(t, -0.01);
        let Middle(t) = long.into();
        assert_relative_eq!(t, -0.6);
        let Right(t) = long.into();
        assert_relative_eq!(t, -36.0);

        let long: LongAngle = Left(-0.01).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, -36.0, epsilon = EPS);

        let long: LongAngle = Middle(-0.6).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, -36.0, epsilon = EPS);

        let long: LongAngle = Right(-36.0).into();
        assert_eq!(long.0, 0);
        assert_eq!(long.1, 0);
        assert_relative_eq!(long.2, -36.0, epsilon = EPS);

        let mut rng = thread_rng();
        let side = Uniform::new(-360.0_f64, 360.0_f64);

        let left = Uniform::new(-360i32, 360i32);
        let middle = Uniform::new(0i8, 60i8);
        let right = Uniform::new(0.0_f64, 60.0_f64);

        for _ in 0..ITERATIONS {
            let left_value = rng.sample(side);

            let long: LongAngle = Left(left_value).into();
            let Left(t) = long.into();
            assert_relative_eq!(t, left_value);

            let value1 = rng.sample(left);
            let value2 = rng.sample(middle);
            let value3 = rng.sample(right);

            let long = LongAngle(value1, value2, value3);
            let value: Left = long.into();
            let LongAngle(t1, t2, t3) = value.into();
            assert_eq!(t1, value1);
            assert_eq!(t2, value2);
            assert_relative_eq!(t3, value3, epsilon = EPS);

            let middle_value = 60.0 * left_value;

            let long: LongAngle = Middle(middle_value).into();
            let Middle(t) = long.into();
            assert_relative_eq!(t, middle_value);
            let Left(t) = long.into();
            assert_relative_eq!(t, left_value);

            let long = LongAngle(value1, value2, value3);
            let value: Middle = long.into();
            let LongAngle(t1, t2, t3) = value.into();
            assert_eq!(t1, value1);
            assert_eq!(t2, value2);
            assert_relative_eq!(t3, value3, epsilon = EPS);

            let right_value = 60.0 * middle_value;

            let long: LongAngle = Right(right_value).into();
            let Right(t) = long.into();
            assert_relative_eq!(t, right_value);
            let Middle(t) = long.into();
            assert_relative_eq!(t, middle_value);
            let Left(t) = long.into();
            assert_relative_eq!(t, left_value);

            let long = LongAngle(value1, value2, value3);
            let value: Right = long.into();
            let LongAngle(t1, t2, t3) = value.into();
            assert_eq!(t1, value1);
            assert_eq!(t2, value2);
            assert_relative_eq!(t3, value3, epsilon = EPS);
        }
    }
}