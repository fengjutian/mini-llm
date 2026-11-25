//! IEEE 754 half-precision conversion (storage only; compute stays `f32`).
//!
//! Used by the checkpoint writer so a model can be stored at half the size.
//! Values are rounded to nearest-even, and subnormals, infinities and NaNs round
//! trip like the hardware conversion.

/// Converts an `f32` to its `f16` bit pattern.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        // Infinity or NaN.
        let payload = if mantissa == 0 {
            0
        } else {
            0x200 | ((mantissa >> 13) as u16 & 0x3ff)
        };
        return sign | 0x7c00 | payload;
    }

    let exp = exponent - 127 + 15;
    if exp >= 0x1f {
        // Overflow to infinity.
        return sign | 0x7c00;
    }

    if exp <= 0 {
        // Subnormal or zero.
        if exp < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (1 - exp) as u32 + 13;
        let mut half = (mantissa >> shift) as u16;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half & 1) == 1) {
            half = half.wrapping_add(1);
        }
        return sign | half;
    }

    let mut half = (((exp as u32) << 10) | (mantissa >> 13)) as u16;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
        half = half.wrapping_add(1);
    }
    sign | half
}

/// Converts an `f16` bit pattern back to `f32`.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Normalize the subnormal.
        let mut m = mantissa;
        let mut shift = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            shift += 1;
        }
        let exponent = 127 - 14 - shift;
        let fraction = (m & 0x3ff) << 13;
        return f32::from_bits(sign | ((exponent as u32) << 23) | fraction);
    }

    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }

    let exponent = exponent + (127 - 15);
    f32::from_bits(sign | (exponent << 23) | (mantissa << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_roundtrip() {
        for (value, bits) in [
            (0.0f32, 0x0000u16),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
        ] {
            assert_eq!(f32_to_f16(value), bits, "encoding {value}");
        }
    }

    #[test]
    fn subnormals_round_trip() {
        let smallest = 2.0f32.powi(-24);
        assert_eq!(f32_to_f16(smallest), 0x0001);
        assert_eq!(f16_to_f32(0x0001), smallest);
        assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0x0000);
    }

    #[test]
    fn general_values_are_close() {
        for value in [0.1f32, -3.25, 1234.5, 1e-3, 6.5e-5] {
            let roundtrip = f16_to_f32(f32_to_f16(value));
            let tolerance = value.abs() * 1e-3 + 1e-7;
            assert!(
                (roundtrip - value).abs() <= tolerance,
                "{value} -> {roundtrip}"
            );
        }
    }

    #[test]
    fn nan_stays_nan() {
        let nan = f16_to_f32(f32_to_f16(f32::NAN));
        assert!(nan.is_nan());
    }
}
