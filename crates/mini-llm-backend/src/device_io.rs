//! Save/load device tensors and checkpoints.
//!
//! Version 1 stores tensors only; version 2 prepends a UTF-8 metadata blob
//! (hyperparameters, corpus fingerprint) so a checkpoint is self-describing.
//! Device tensors are copied to the host for writing and uploaded to the
//! requested device on load.

use std::path::Path;

use mini_llm_tensor::Tensor;

use crate::{BackendError, Device, DeviceTensor};

const MAGIC: &[u8; 4] = b"DEVT";
const VERSION_TENSORS: u32 = 1;
const VERSION_CHECKPOINT: u32 = 2;

/// Writes `tensors` to `path` (no metadata).
pub fn save_device_tensors(
    path: impl AsRef<Path>,
    tensors: &[DeviceTensor],
) -> Result<(), BackendError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION_TENSORS.to_le_bytes());
    push_tensors(&mut bytes, tensors)?;
    std::fs::write(path, bytes).map_err(|error| BackendError::Backend(error.to_string()))
}

/// Writes `tensors` plus a metadata blob (version 2).
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
    std::fs::write(path, bytes).map_err(|error| BackendError::Backend(error.to_string()))
}

/// Loads tensors from `path`, ignoring any metadata.
pub fn load_device_tensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<Vec<DeviceTensor>, BackendError> {
    Ok(load_device_checkpoint(path, device)?.0)
}

/// Loads tensors and optional metadata from `path`.
pub fn load_device_checkpoint(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<(Vec<DeviceTensor>, Option<String>), BackendError> {
    let bytes = std::fs::read(path).map_err(|error| BackendError::Backend(error.to_string()))?;
    let mut reader = Reader {
        bytes: &bytes,
        position: 0,
    };
    if reader.take(4)? != MAGIC {
        return Err(BackendError::Backend("not a device tensor file".into()));
    }
    let version = reader.u32()?;
    let metadata = match version {
        VERSION_CHECKPOINT => {
            let length = reader.u32()? as usize;
            let bytes = reader.take(length)?;
            Some(
                std::str::from_utf8(bytes)
                    .map_err(|error| BackendError::Backend(error.to_string()))?
                    .to_string(),
            )
        }
        VERSION_TENSORS => None,
        other => {
            return Err(BackendError::Backend(format!(
                "unsupported device tensor version {other}"
            )));
        }
    };
    let count = reader.u32()? as usize;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = reader.u32()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(reader.u32()? as usize);
        }
        let length: usize = shape.iter().product();
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(reader.f32()?);
        }
        let host = Tensor::new(data, shape)?;
        tensors.push(DeviceTensor::from_host(host).to(device)?);
    }
    Ok((tensors, metadata))
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

    fn u32(&mut self) -> Result<u32, BackendError> {
        let bytes = self.take(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(buffer))
    }

    fn f32(&mut self) -> Result<f32, BackendError> {
        let bytes = self.take(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(bytes);
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
        let (loaded, metadata) = load_device_checkpoint(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to_host().unwrap().data(), [1.5, 2.5]);
        assert_eq!(metadata.as_deref(), Some("layers=2 hidden=32 seq=64"));
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
