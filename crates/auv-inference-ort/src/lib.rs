use auv_inference_common::{InferenceError, InferenceResult};
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "runtime")]
use ort::{session::Session, value::TensorRef};
use std::{
  path::{Path, PathBuf},
  sync::Mutex,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrtModelConfig {
  pub model_path: PathBuf,
  pub execution_provider: ExecutionProvider,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionProvider {
  #[default]
  Cpu,
  CoreMl,
  Cuda,
  DirectMl,
  OpenVino,
  TensorRt,
  WebGpu,
  Xnnpack,
}

#[derive(Clone, Debug, PartialEq)]
pub struct F32Tensor {
  pub name: String,
  pub shape: Vec<usize>,
  pub data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopPrediction {
  pub index: usize,
  pub confidence: f32,
}

#[cfg(feature = "runtime")]
pub struct OrtSession {
  model: Mutex<Session>,
}

#[cfg(feature = "runtime")]
impl std::fmt::Debug for OrtSession {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter.debug_struct("OrtSession").finish_non_exhaustive()
  }
}

#[cfg(not(feature = "runtime"))]
pub struct OrtSession;

#[cfg(feature = "runtime")]
impl OrtSession {
  pub fn load(config: OrtModelConfig) -> InferenceResult<Self> {
    require_model_path(&config.model_path)?;

    let mut builder = Session::builder().map_err(backend_error)?;
    let providers = execution_providers(config.execution_provider)?;
    builder = builder.with_execution_providers(providers).map_err(backend_error)?;
    let model = builder.commit_from_file(&config.model_path).map_err(backend_error)?;

    Ok(Self {
      model: Mutex::new(model),
    })
  }

  pub fn run_f32(&self, input: F32Tensor) -> InferenceResult<Vec<F32Tensor>> {
    self.run_f32_many(vec![input])
  }

  pub fn run_f32_many(&self, inputs: Vec<F32Tensor>) -> InferenceResult<Vec<F32Tensor>> {
    let arrays = inputs
      .into_iter()
      .map(|input| {
        let array = ArrayD::from_shape_vec(IxDyn(&input.shape), input.data).map_err(|error| InferenceError::Backend {
          message: error.to_string(),
        })?;
        Ok((input.name, array))
      })
      .collect::<InferenceResult<Vec<_>>>()?;
    let tensors = arrays
      .iter()
      .map(|(name, array)| Ok((name.as_str(), TensorRef::from_array_view(array.view()).map_err(backend_error)?)))
      .collect::<InferenceResult<Vec<_>>>()?;
    let mut model = self.model.lock().map_err(|error| InferenceError::SessionUnavailable {
      reason: error.to_string(),
    })?;
    let outputs = model.run(tensors).map_err(backend_error)?;

    outputs
      .keys()
      .map(|name| {
        let value = outputs
          .get(name)
          .ok_or_else(|| InferenceError::Backend {
            message: format!("missing ORT output value for {name}"),
          })?
          .try_extract_tensor::<f32>()
          .map_err(backend_error)?;
        Ok(F32Tensor {
          name: name.to_owned(),
          shape: value.0.iter().map(|dim| *dim as usize).collect(),
          data: value.1.to_vec(),
        })
      })
      .collect()
  }
}

#[cfg(not(feature = "runtime"))]
impl OrtSession {
  pub fn load(config: OrtModelConfig) -> InferenceResult<Self> {
    require_model_path(&config.model_path)?;
    Err(InferenceError::Backend {
      message: "auv-inference-ort built without runtime feature".to_string(),
    })
  }
}

pub fn provider_name(provider: ExecutionProvider) -> &'static str {
  match provider {
    ExecutionProvider::Cpu => "CPUExecutionProvider",
    ExecutionProvider::CoreMl => "CoreMLExecutionProvider",
    ExecutionProvider::Cuda => "CUDAExecutionProvider",
    ExecutionProvider::DirectMl => "DmlExecutionProvider",
    ExecutionProvider::OpenVino => "OpenVINOExecutionProvider",
    ExecutionProvider::TensorRt => "TensorrtExecutionProvider",
    ExecutionProvider::WebGpu => "WebGPUExecutionProvider",
    ExecutionProvider::Xnnpack => "XnnpackExecutionProvider",
  }
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
  let Some(max) = logits.iter().copied().reduce(f32::max) else {
    return Vec::new();
  };
  let exp = logits.iter().map(|value| (*value - max).exp()).collect::<Vec<_>>();
  let sum = exp.iter().sum::<f32>();
  if sum == 0.0 || !sum.is_finite() {
    return vec![0.0; logits.len()];
  }
  exp.into_iter().map(|value| value / sum).collect()
}

pub fn top1(values: &[f32]) -> Option<TopPrediction> {
  values
    .iter()
    .copied()
    .enumerate()
    .max_by(|(_, left), (_, right)| left.total_cmp(right))
    .map(|(index, confidence)| TopPrediction { index, confidence })
}

fn require_model_path(path: &Path) -> InferenceResult<()> {
  if path.exists() {
    Ok(())
  } else {
    Err(InferenceError::MissingModel {
      path: path.to_path_buf(),
    })
  }
}

#[cfg(feature = "runtime")]
fn backend_error<R>(error: ort::Error<R>) -> InferenceError {
  InferenceError::Backend {
    message: error.to_string(),
  }
}

#[cfg(feature = "runtime")]
fn execution_providers(provider: ExecutionProvider) -> InferenceResult<Vec<ort::ep::ExecutionProviderDispatch>> {
  #[allow(unreachable_patterns)]
  let providers = match provider {
    ExecutionProvider::Cpu => vec![ort::ep::CPU::default().build().error_on_failure()],
    #[cfg(feature = "coreml")]
    ExecutionProvider::CoreMl => vec![ort::ep::CoreML::default().build().error_on_failure()],
    #[cfg(feature = "cuda")]
    ExecutionProvider::Cuda => vec![ort::ep::CUDA::default().build().error_on_failure()],
    #[cfg(feature = "directml")]
    ExecutionProvider::DirectMl => vec![ort::ep::DirectML::default().build().error_on_failure()],
    #[cfg(feature = "openvino")]
    ExecutionProvider::OpenVino => vec![ort::ep::OpenVINO::default().build().error_on_failure()],
    #[cfg(feature = "tensorrt")]
    ExecutionProvider::TensorRt => vec![ort::ep::TensorRT::default().build().error_on_failure()],
    #[cfg(feature = "webgpu")]
    ExecutionProvider::WebGpu => vec![ort::ep::WebGPU::default().build().error_on_failure()],
    #[cfg(feature = "xnnpack")]
    ExecutionProvider::Xnnpack => vec![ort::ep::XNNPACK::default().build().error_on_failure()],
    _ => {
      return Err(InferenceError::Backend {
        message: format!(
          "requested execution provider {} is unavailable because auv-inference-ort was built without the '{}' feature",
          provider_name(provider),
          provider_feature(provider)
        ),
      });
    }
  };
  Ok(providers)
}

#[cfg(feature = "runtime")]
fn provider_feature(provider: ExecutionProvider) -> &'static str {
  match provider {
    ExecutionProvider::Cpu => "runtime",
    ExecutionProvider::CoreMl => "coreml",
    ExecutionProvider::Cuda => "cuda",
    ExecutionProvider::DirectMl => "directml",
    ExecutionProvider::OpenVino => "openvino",
    ExecutionProvider::TensorRt => "tensorrt",
    ExecutionProvider::WebGpu => "webgpu",
    ExecutionProvider::Xnnpack => "xnnpack",
  }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
