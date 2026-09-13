use auv_inference_common::InferenceError;
use std::path::PathBuf;

#[cfg(feature = "runtime")]
use crate::execution_providers;
use crate::{ExecutionProvider, OrtModelConfig, OrtSession, TopPrediction, provider_name, softmax, top1};

#[test]
fn missing_model_is_rejected_before_backend_load() {
  let path = PathBuf::from("missing-card-corner-classifier.onnx");
  let err = OrtSession::load(OrtModelConfig {
    model_path: path.clone(),
    execution_provider: ExecutionProvider::Cpu,
  })
  .unwrap_err();

  assert!(matches!(err, InferenceError::MissingModel { path: actual } if actual == path));
}

#[test]
fn provider_names_match_onnx_runtime_identifiers() {
  assert_eq!(provider_name(ExecutionProvider::Cpu), "CPUExecutionProvider");
  assert_eq!(provider_name(ExecutionProvider::CoreMl), "CoreMLExecutionProvider");
  assert_eq!(provider_name(ExecutionProvider::Cuda), "CUDAExecutionProvider");
}

#[cfg(all(feature = "runtime", not(feature = "coreml")))]
#[test]
fn coreml_without_its_feature_is_rejected() {
  let error = execution_providers(ExecutionProvider::CoreMl).expect_err("CoreML must not fall back to CPU");

  assert!(matches!(
    error,
    InferenceError::Backend { message }
      if message == "requested execution provider CoreMLExecutionProvider is unavailable because auv-inference-ort was built without the 'coreml' feature"
  ));
}

#[cfg(all(feature = "runtime", not(feature = "cuda")))]
#[test]
fn cuda_without_its_feature_is_rejected() {
  let error = execution_providers(ExecutionProvider::Cuda).expect_err("CUDA must not fall back to CPU");

  assert!(matches!(
    error,
    InferenceError::Backend { message }
      if message == "requested execution provider CUDAExecutionProvider is unavailable because auv-inference-ort was built without the 'cuda' feature"
  ));
}

#[test]
fn softmax_returns_normalized_probabilities() {
  let probabilities = softmax(&[1.0, 2.0, 3.0]);
  let total = probabilities.iter().sum::<f32>();

  assert!((total - 1.0).abs() < 1e-6);
  assert!(probabilities[2] > probabilities[1]);
  assert!(probabilities[1] > probabilities[0]);
}

#[test]
fn top1_returns_index_and_confidence() {
  assert_eq!(
    top1(&[0.1, 0.8, 0.3]),
    Some(TopPrediction {
      index: 1,
      confidence: 0.8
    })
  );
  assert_eq!(top1(&[]), None);
}
