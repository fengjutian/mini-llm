//! Save/load device tensors and checkpoints.
//!
//! Version 1 stores tensors only; version 2 prepends a UTF-8 metadata blob
//! (hyperparameters, corpus fingerprint) so a checkpoint is self-describing.
//! Version 3 adds a per-tensor storage format (f32 / f16 / symmetric int8), so
//! checkpoints can be written quantized and are dequantized to f32 on load.
//! Device tensors are copied to the host for writing and uploaded to the
//! requested device on load.

use std::path::Path;

use half::f16;
use mini_llm_tensor::Tensor;

use crate::{BackendError, Device, DeviceTensor};

const MAGIC: &[u8; 4] = b"DEVT";
const VERSION_TENSORS: u32 = 1;
const VERSION_CHECKPOINT: u32 = 2;
const VERSION_QUANTIZED: u32 = 3;

/// How tensor values are stored inside a checkpoint file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    /// Raw 32-bit floats (the compute format).
    F32,
    /// IEEE 754 half precision.
    F16,
    /// Symmetric per-tensor int8 with an f32 scale.
    Q8,
}

impl TensorFormat {
    /// Short name used by the CLI (`f32`, `f16`, `q8`).
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8 => "q8",
        }
    }

    /// Parses a storage format name.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "q8" => Some(Self::Q8),
            _ => None,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8 => 2,
        }
    }

    fn from_code(code: u8) -> Result<Self, BackendError> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q8),
            other => Err(BackendError::Backend(format!(
                "unsupported tensor storage code {other}"
            ))),
        }
    }
}

/// Writes `tensors` to `path` (no metadata).
pub fn save_device_tensors(
    path: impl AsRef<Path>,
    tensors: &[DeviceTensor],
) -> Result<(), BackendError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION_TENSORS.to_le_bytes());
    push_tensors(&mut bytes, tensors)?;
    write_bytes(path, &bytes)
}

/// Writes `tensors` plus a metadata blob (version 2, always f32).
pub fn save_device_checkpoint(
    path: impl AsRef<Path>,
    tensors: &[DeviceTensor],
    metadata: &str,
) -> Result<(), BackendError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION_CHECKPOINT.to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    bytes.extend_from_slice(metadata.as_bytes());
    push_tensors(&mut bytes, tensors)?;
    write_bytes(path, &bytes)
}

/// Writes `tensors` with metadata and an explicit storage format (version 3).
///
/// Loading dequantizes f16 / q8 back to f32, so training code is unchanged.
pub fn save_device_checkpoint_with_format(
    path: impl AsRef<Path>,
    tensors: &[DeviceTensor],
    metadata: &str,
    format: TensorFormat,
) -> Result<(), BackendError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION_QUANTIZED.to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    bytes.extend_from_slice(metadata.as_bytes());
    push_tensors_with_format(&mut bytes, tensors, format)?;
    write_bytes(path, &bytes)
}

/// Loads tensors from `path`, ignoring any metadata.
pub fn load_device_tensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<Vec<DeviceTensor>, BackendError> {
    Ok(load_device_checkpoint(path, device)?.0)
}

/// Loads tensors, optional metadata and the stored format from `path`.
pub fn load_device_checkpoint(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<(Vec<DeviceTensor>, Option<String>, TensorFormat), BackendError> {
    let bytes = std::fs::read(path).map_err(|error| BackendError::Backend(error.to_string()))?;
    let mut reader = Reader {
        bytes: &bytes,
        position: 0,
    };
    if reader.take(4)? != MAGIC {
        return Err(BackendError::Backend("not a device tensor file".into()));
    }
    let version = reader.u32()?;
    let (metadata, quantized) = match version {
        VERSION_CHECKPOINT => {
            let length = reader.u32()? as usize;
            let bytes = reader.take(length)?;
            (
                Some(
                    std::str::from_utf8(bytes)
                        .map_err(|error| BackendError::Backend(error.to_string()))?
                        .to_string(),
                ),
                false,
            )
        }
        VERSION_QUANTIZED => {
            let length = reader.u32()? as usize;
            let bytes = reader.take(length)?;
            (
                Some(
                    std::str::from_utf8(bytes)
                        .map_err(|error| BackendError::Backend(error.to_string()))?
                        .to_string(),
                ),
                true,
            )
        }
        VERSION_TENSORS => (None, false),
        other => {
            return Err(BackendError::Backend(format!(
                "unsupported device tensor version {other}"
            )));
        }
    };

    let count = reader.u32()? as usize;
    let mut tensors = Vec::with_capacity(count);
    let mut format = TensorFormat::F32;
    for index in 0..count {
        let tensor_format = if quantized {
            TensorFormat::from_code(reader.u8()?)?
        } else {
            TensorFormat::F32
        };
        if index == 0 {
            format = tensor_format;
        } else if format != tensor_format {
            return Err(BackendError::Backend(
                "checkpoint mixes tensor storage formats".into(),
            ));
        }
        let rank = reader.u32()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(reader.u32()? as usize);
        }
        let length: usize = shape.iter().product();
        let data = match tensor_format {
            TensorFormat::F32 => {
                let mut data = Vec::with_capacity(length);
                for _ in 0..length {
                    data.push(reader.f32()?);
                }
                data
            }
            TensorFormat::F16 => {
                let mut data = Vec::with_capacity(length);
                for _ in 0..length {
                    data.push(f16::from_bits(reader.u16()?).to_f32());
                }
                data
            }
            TensorFormat::Q8 => {
                let scale = reader.f32()?;
                let mut data = Vec::with_capacity(length);
                for _ in 0..length {
                    data.push(f32::from(reader.i8()?) * scale);
                }
                data
            }
        };
        let host = Tensor::new(data, shape)?;
        tensors.push(DeviceTensor::from_host(host).to(device)?);
    }
    Ok((tensors, metadata, format))
}

fn write_bytes(path: impl AsRef<Path>, bytes: &[u8]) -> Result<(), BackendError> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
        }
    }
    std::fs::write(path, bytes).map_err(|error| BackendError::Backend(error.to_string()))
}

fn push_tensors(bytes: &mut Vec<u8>, tensors: &[DeviceTensor]) -> Result<(), BackendError> {
    bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
        let host = tensor.to_host()?;
        let shape = host.shape();
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dimension in shape {
            bytes.extend_from_slice(&(*dimension as u32).to_le_bytes());
        }
        for value in host.data() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn push_tensors_with_format(
    bytes: &mut Vec<u8>,
    tensors: &[DeviceTensor],
    format: TensorFormat,
) -> Result<(), BackendError> {
    bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
        let host = tensor.to_host()?;
        let shape = host.shape();
        bytes.push(format.code());
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dimension in shape {
            bytes.extend_from_slice(&(*dimension as u32).to_le_bytes());
        }
        match format {
            TensorFormat::F32 => {
                for value in host.data() {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            TensorFormat::F16 => {
                for value in host.data() {
                    bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
                }
            }
            TensorFormat::Q8 => {
                let max_abs = host
                    .data()
                    .iter()
                    .fold(0.0f32, |accumulator, value| accumulator.max(value.abs()));
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                bytes.extend_from_slice(&scale.to_le_bytes());
                for value in host.data() {
                    let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
                    bytes.push(quantized as u8);
                }
            }
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn take(&mut self, length: usize) -> Result<&[u8], BackendError> {
        let end = self.position + length;
        if end > self.bytes.len() {
            return Err(BackendError::Backend("truncated device tensor file".into()));
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, BackendError> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, BackendError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, BackendError> {
        let mut buffer = [0u8; 2];
        buffer.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(buffer))
    }

    fn u32(&mut self) -> Result<u32, BackendError> {
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buffer))
    }

    fn f32(&mut self) -> Result<f32, BackendError> {
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(self.take(4)?);
        Ok(f32::from_le_bytes(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_host_tensors() {
        let path = std::env::temp_dir().join("mini_llm_device_io_test.bin");
        let first = Tensor::from_2d(2, 2, vec![1.0, -2.0, 3.5, 0.0]).unwrap();
        let second = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]).unwrap();
        let tensors = [
            DeviceTensor::from_host(first.clone()),
            DeviceTensor::from_host(second.clone()),
        ];
        save_device_tensors(&path, &tensors).unwrap();
        let loaded = load_device_tensors(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].to_host().unwrap().data(), first.data());
        assert_eq!(loaded[0].shape(), first.shape());
        assert_eq!(loaded[1].to_host().unwrap().data(), second.data());
        assert_eq!(loaded[1].shape(), second.shape());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn roundtrips_checkpoint_metadata() {
        let path = std::env::temp_dir().join("mini_llm_device_io_meta.bin");
        let tensor = Tensor::from_2d(1, 2, vec![1.5, 2.5]).unwrap();
        let tensors = [DeviceTensor::from_host(tensor)];
        save_device_checkpoint(&path, &tensors, "layers=2 hidden=32 seq=64").unwrap();
        let (loaded, metadata, format) = load_device_checkpoint(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to_host().unwrap().data(), [1.5, 2.5]);
        assert_eq!(metadata.as_deref(), Some("layers=2 hidden=32 seq=64"));
        assert_eq!(format, TensorFormat::F32);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn f16_checkpoint_roundtrips_within_half_precision() {
        let path = std::env::temp_dir().join("mini_llm_device_io_f16.bin");
        let values = vec![0.5, -1.25, 3.0, 0.001];
        let tensor = Tensor::new(values.clone(), vec![2, 2]).unwrap();
        let tensors = [DeviceTensor::from_host(tensor)];
        save_device_checkpoint_with_format(&path, &tensors, "f16", TensorFormat::F16).unwrap();
        let (loaded, _, format) = load_device_checkpoint(&path, &Device::Cpu).unwrap();
        assert_eq!(format, TensorFormat::F16);
        for (expected, actual) in values.iter().zip(loaded[0].to_host().unwrap().data()) {
            let tolerance = expected.abs() * 1e-3 + 1e-6;
            assert!(
                (expected - actual).abs() <= tolerance,
                "f16 {actual} vs {expected}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_checkpoint_roundtrips_within_one_step() {
        let path = std::env::temp_dir().join("mini_llm_device_io_q8.bin");
        let values = vec![0.0, 0.25, -0.5, 1.0, -1.0, 0.125];
        let tensor = Tensor::new(values.clone(), vec![2, 3]).unwrap();
        let tensors = [DeviceTensor::from_host(tensor)];
        save_device_checkpoint_with_format(&path, &tensors, "q8", TensorFormat::Q8).unwrap();
        let (loaded, _, format) = load_device_checkpoint(&path, &Device::Cpu).unwrap();
        assert_eq!(format, TensorFormat::Q8);
        let step = 1.0 / 127.0;
        for (expected, actual) in values.iter().zip(loaded[0].to_host().unwrap().data()) {
            assert!(
                (expected - actual).abs() <= step,
                "q8 {actual} vs {expected}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("mini_llm_device_io_bad.bin");
        std::fs::write(&path, b"NOPE").unwrap();
        assert!(load_device_tensors(&path, &Device::Cpu).is_err());
        let _ = std::fs::remove_file(path);
    }
}
