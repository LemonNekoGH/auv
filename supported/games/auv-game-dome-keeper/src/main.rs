use std::{
  collections::VecDeque,
  error::Error,
  path::{Path, PathBuf},
  thread,
  time::{Duration, Instant},
};

#[cfg(not(target_os = "macos"))]
compile_error!("auv-game-dome-keeper supports only macOS with CoreML");

use auv_api_proto::auv::api::{
  driver::v1 as frame_buffer_proto,
  image::v1::{RgbFrame, RgbaFrame},
};
use auv_driver::{
  Activation, App, CaptureOptions, Click, InputPolicy, InputTarget, LocalDriverSession, PermissionStatus, PressKeysOptions, Window,
  WindowPoint, WindowSelector,
};
use auv_inference_ort::{ExecutionProvider, F32Tensor, OrtModelConfig, OrtSession, provider_name, softmax, top1};
use clap::Parser;
use image::RgbaImage;

const FRAME_COUNT: usize = 10;
const MODEL_WIDTH: u32 = 384;
const MODEL_HEIGHT: u32 = 216;
const TITLE_BAR_HEIGHT: u32 = 28;
const MODEL_FRAME_BYTES: usize = MODEL_WIDTH as usize * MODEL_HEIGHT as usize * 3;
const ACTIONS: [&str; 9] = [
  "ui_up",
  "ui_down",
  "ui_left",
  "ui_right",
  "ui_select",
  "keeper1_pickup",
  "keeper1_drop",
  "dome1_fire",
  "none",
];
const TASKS: [&str; 5] = ["pickup", "drop", "activate", "attack", "enter"];
const TARGETS: [&str; 6] = [
  "iron",
  "cobalt",
  "water",
  "gadget_chamber",
  "monster",
  "mine",
];
const NONE: usize = 8;

#[derive(Debug, Parser)]
#[command(about = "Run the AUV capture -> Lower v0 ONNX -> input loop for Dome Keeper")]
struct Args {
  #[arg(long)]
  model: PathBuf,

  #[arg(long, default_value = "Godot")]
  app: String,

  #[arg(long, default_value = "Dome Keeper (DEBUG)")]
  title_contains: String,

  #[arg(long)]
  frames_dir: PathBuf,

  #[arg(long, default_value = "pickup")]
  task: String,

  #[arg(long, default_value = "iron")]
  target: String,

  #[arg(long, default_value_t = 8)]
  steps: u32,

  #[arg(long, default_value_t = 500)]
  settle_ms: u64,

  #[arg(long, default_value_t = 10)]
  target_fps: u32,

  #[arg(long, default_value_t = 10)]
  buffer_capacity: u32,

  #[arg(long, default_value_t = 2_000)]
  frame_wait_timeout_ms: u32,

  /// Activate the app, click and verify the target window, then deliver
  /// global input. Useful when the window is absent from its AX window tree.
  #[arg(long)]
  foreground_input: bool,

  /// Deliver keyboard events to the captured window's process without AX focus.
  #[arg(long, conflicts_with = "foreground_input")]
  background_input: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  let args = Args::parse();
  let instruction = instruction(&args.task, &args.target)?;
  std::fs::create_dir_all(&args.frames_dir)?;

  let auv = auv::Client::from_env().await?;
  let run = auv.run(Default::default()).await?;
  let result = run_controller(&run, &args, &instruction).await;
  let outcome = if result.is_ok() {
    auv::runs::RunOutcome::Succeeded
  } else {
    auv::runs::RunOutcome::Failed
  };
  let finish = run.finish_if_owned(outcome).await.map(|_| ()).map_err(Box::<dyn Error>::from);
  merge_result(result, finish)
}

async fn run_controller(run: &auv::client::RunClient, args: &Args, instruction: &[f32]) -> Result<(), Box<dyn Error>> {
  let session = auv_driver::open_local()?;
  let permissions = session.permission().probe()?;
  println!("permissions: accessibility={:?} screen_capture={:?}", permissions.accessibility, permissions.screen_capture_kit);
  if permissions.accessibility != PermissionStatus::Granted {
    return Err("the auv-game-dome-keeper process needs macOS Accessibility permission before it can send input".into());
  }
  if permissions.screen_capture_kit != PermissionStatus::Granted {
    return Err("the auv-game-dome-keeper process needs macOS Screen Recording permission before it can capture frames".into());
  }
  let window = session.window().resolve(
    WindowSelector {
      app: Some(App::name(&args.app)),
      main_visible: true,
      ..WindowSelector::default()
    }
    .title_contains(&args.title_contains),
  )?;
  let execution_provider = execution_provider();
  let execution_provider_name = provider_name(execution_provider);
  let model = OrtSession::load(OrtModelConfig {
    model_path: args.model.clone(),
    execution_provider,
  })?;

  let frame_buffer_runner = run.runner(Default::default()).await?;
  let mut frame_buffer =
    frame_buffer_proto::recent_frames_service_client::RecentFramesServiceClient::new(frame_buffer_runner.extension_transport()?)
      .max_decoding_message_size(auv::client::runner::IMAGE_RPC_MESSAGE_SIZE_LIMIT)
      .max_encoding_message_size(auv::client::runner::IMAGE_RPC_MESSAGE_SIZE_LIMIT);
  let opened = frame_buffer
    .open_frame_buffer(frame_buffer_proto::OpenFrameBufferRequest {
      target: Some(frame_buffer_proto::CaptureTarget {
        target: Some(frame_buffer_proto::capture_target::Target::Window(frame_buffer_proto::WindowRef {
          window_id: window.reference.id.clone(),
        })),
      }),
      target_fps: args.target_fps,
      frame_capacity: args.buffer_capacity,
      output_size: None,
    })
    .await?
    .into_inner();
  let frame_buffer_ref = opened.frame_buffer.ok_or("OpenFrameBuffer response omitted frame_buffer")?;

  println!("target: {}", window.title.as_deref().unwrap_or("untitled window"));
  println!("instruction: {} {}", args.task, args.target);
  println!("execution_provider: {execution_provider_name}");
  println!("frame_buffer: {}", frame_buffer_ref.frame_buffer_id);

  let loop_result: Result<(), Box<dyn Error>> = async {
    let before = if args.foreground_input {
      session.window().capture_with(
        &window,
        CaptureOptions {
          activation: Activation::ActivateFirst {
            settle: Duration::from_millis(100),
          },
          ..CaptureOptions::default()
        },
      )?
    } else {
      session.window().capture(&window)?
    }
    .image;
    save_evidence_frame(&args.frames_dir, "before", &before)?;
    if args.foreground_input {
      focus_by_click(&session, &window, &args.title_contains)?;
    }
    let input_target = if args.foreground_input {
      InputTarget::Foreground
    } else {
      InputTarget::Window(window.clone())
    };
    let input_policy = if args.background_input {
      InputPolicy::BackgroundPreferred
    } else {
      InputPolicy::ForegroundPreferred
    };
    let mut held = VecDeque::from(vec![NONE; FRAME_COUNT]);
    let mut frame_history = VecDeque::new();
    let mut trace = Vec::new();
    let mut after_sequence = 0;

    for step in 0..args.steps {
      let frame_rpc_started = Instant::now();
      let deadline = Instant::now() + Duration::from_millis(u64::from(args.frame_wait_timeout_ms));
      let response = loop {
        let response = frame_buffer
          .get_recent_frames(frame_buffer_proto::GetRecentFramesRequest {
            frame_buffer: Some(frame_buffer_ref.clone()),
            after_sequence,
          })
          .await?
          .into_inner();
        if !response.frames.is_empty() {
          break response;
        }
        if Instant::now() >= deadline {
          return Err(format!("GetRecentFrames timed out after {} ms without a newer frame", args.frame_wait_timeout_ms).into());
        }
        tokio::time::sleep(Duration::from_secs_f64(1.0 / f64::from(args.target_fps))).await;
      };
      let frame_rpc_ms = frame_rpc_started.elapsed().as_millis();
      let latest_sequence = response.latest_sequence;
      let (frames, first_sequence, last_sequence) =
        model_history(&mut frame_history, response.frames, after_sequence, latest_sequence)?;
      after_sequence = latest_sequence;

      let inference_started = Instant::now();
      let logits = infer(&model, &frames, &held, instruction)?;
      let inference_ms = inference_started.elapsed().as_millis();
      let probabilities = softmax(&logits);
      let prediction = top1(&probabilities).ok_or("model returned no logits")?;
      let action = ACTIONS[prediction.index];

      let input_started = Instant::now();
      if let Some(key) = action_key(prediction.index) {
        if args.foreground_input {
          require_frontmost(&session, &window, &args.title_contains)?;
        }
        session.input().press_keys(
          &input_target,
          PressKeysOptions {
            keys: vec![key.to_owned()],
            ..PressKeysOptions::default()
          },
          input_policy,
          false,
        )?;
      }
      let input_ms = input_started.elapsed().as_millis();

      println!(
        "step={step} action={action} confidence={:.3} frame_rpc_ms={frame_rpc_ms} inference_ms={inference_ms} input_ms={input_ms} sequences={first_sequence}..={last_sequence}",
        prediction.confidence
      );
      trace.push(format!(
        "Lower v0: {} {} | provider: {execution_provider_name} | model frames: RPC sequences {first_sequence}..={last_sequence}, consumer cropped top {TITLE_BAR_HEIGHT} px and prepared {MODEL_WIDTH}x{MODEL_HEIGHT} RGB8 | predicted: {} ({:.1}%) | AUV sent: {}",
        args.task,
        args.target,
        action,
        prediction.confidence * 100.0,
        action_key(prediction.index).unwrap_or("nothing")
      ));

      held.pop_front();
      held.push_back(NONE);
      tokio::time::sleep(Duration::from_millis(args.settle_ms)).await;
    }

    let final_capture = session.window().capture(&window)?;
    save_evidence_frame(&args.frames_dir, "after", &final_capture.image)?;
    trace.push("Final AUV evidence observation".to_owned());
    write_subtitles(&args.frames_dir.join("trace.srt"), &trace)?;
    println!("saved before/after AUV evidence frames and trace.srt");
    Ok(())
  }
  .await;

  let close_result = frame_buffer
    .close_frame_buffer(frame_buffer_proto::CloseFrameBufferRequest {
      frame_buffer: Some(frame_buffer_ref),
    })
    .await
    .map(|_| ())
    .map_err(Box::<dyn Error>::from);
  merge_result(loop_result, close_result)
}

fn execution_provider() -> ExecutionProvider {
  ExecutionProvider::CoreMl
}

fn focus_by_click(session: &LocalDriverSession, window: &Window, title_contains: &str) -> Result<(), Box<dyn Error>> {
  let previous_pointer = session.input().current_position()?;
  let focus_point =
    session.window().to_screen_point(window, WindowPoint::new(window.frame.size.width / 2.0, window.frame.size.height / 2.0))?;
  let focus_result = (|| -> Result<(), Box<dyn Error>> {
    session.input().click_at(focus_point.point(), Click::Single)?;
    thread::sleep(Duration::from_millis(150));
    require_frontmost(session, window, title_contains)
  })();
  let restore_result = session.input().move_to(previous_pointer);
  focus_result?;
  restore_result?;
  println!("focus: target window confirmed frontmost after content click");
  Ok(())
}

fn require_frontmost(session: &LocalDriverSession, expected: &Window, title_contains: &str) -> Result<(), Box<dyn Error>> {
  let frontmost = session.window().resolve(
    WindowSelector {
      app: Some(App::frontmost()),
      main_visible: true,
      ..WindowSelector::default()
    }
    .title_contains(title_contains),
  )?;
  if frontmost.reference != expected.reference {
    return Err(format!("frontmost window changed: expected {}, got {}", expected.reference.id, frontmost.reference.id).into());
  }
  Ok(())
}

fn instruction(task: &str, target: &str) -> Result<Vec<f32>, Box<dyn Error>> {
  let task_index = TASKS.iter().position(|candidate| *candidate == task).ok_or("unknown task")?;
  let target_index = TARGETS.iter().position(|candidate| *candidate == target).ok_or("unknown target")?;
  let valid = matches!(task, "pickup" | "drop") && matches!(target, "iron" | "cobalt" | "water")
    || task == "activate" && target == "gadget_chamber"
    || task == "attack" && target == "monster"
    || task == "enter" && target == "mine";
  if !valid {
    return Err("unsupported task/target pair".into());
  }

  let mut value = vec![0.0; TASKS.len() + TARGETS.len()];
  value[task_index] = 1.0;
  value[TASKS.len() + target_index] = 1.0;
  Ok(value)
}

fn model_history(
  history: &mut VecDeque<(u64, RgbFrame)>,
  recent: Vec<frame_buffer_proto::RecentFrame>,
  after_sequence: u64,
  latest_sequence: u64,
) -> Result<(Vec<RgbFrame>, u64, u64), Box<dyn Error>> {
  if recent.is_empty() {
    return Err("GetRecentFrames returned no frames".into());
  }
  for pair in recent.windows(2) {
    if pair[0].sequence >= pair[1].sequence {
      return Err("GetRecentFrames returned frames outside strict sequence order".into());
    }
  }
  let response_first_sequence = recent.first().expect("non-empty frame response was checked").sequence;
  let response_last_sequence = recent.last().expect("non-empty frame response was checked").sequence;
  if response_first_sequence <= after_sequence {
    return Err(format!("GetRecentFrames returned sequence {response_first_sequence} after caller acknowledged {after_sequence}").into());
  }
  if response_last_sequence != latest_sequence {
    return Err(
      format!("GetRecentFrames latest_sequence {latest_sequence} does not match final frame sequence {response_last_sequence}").into(),
    );
  }

  if after_sequence != 0 && response_first_sequence > after_sequence.saturating_add(1) {
    history.clear();
  }
  for frame in recent.into_iter().map(prepare_model_frame) {
    history.push_back(frame?);
    if history.len() > FRAME_COUNT {
      history.pop_front();
    }
  }

  let first_sequence = history.front().expect("new frames were appended to history").0;
  let last_sequence = history.back().expect("new frames were appended to history").0;
  let first = history.front().expect("new frames were appended to history").1.clone();
  let mut frames = history.iter().map(|(_, frame)| frame.clone()).collect::<Vec<_>>();
  frames.splice(0..0, std::iter::repeat_n(first, FRAME_COUNT - frames.len()));
  Ok((frames, first_sequence, last_sequence))
}

fn prepare_model_frame(recent: frame_buffer_proto::RecentFrame) -> Result<(u64, RgbFrame), Box<dyn Error>> {
  let capture = recent.capture.ok_or("RecentFrame omitted capture")?;
  let frame = capture.image.ok_or("CapturedFrame omitted image")?;
  validate_rgba_frame(recent.sequence, &frame)?;
  if frame.height <= TITLE_BAR_HEIGHT {
    return Err(
      format!("RecentFrame sequence {} is {} pixels high; the model crop removes {TITLE_BAR_HEIGHT} pixels", recent.sequence, frame.height)
        .into(),
    );
  }

  let image = RgbaImage::from_raw(frame.width, frame.height, frame.data)
    .ok_or_else(|| format!("RecentFrame sequence {} could not be decoded as RGBA8", recent.sequence))?;
  let cropped = image::imageops::crop_imm(&image, 0, TITLE_BAR_HEIGHT, image.width(), image.height() - TITLE_BAR_HEIGHT);
  let resized = image::imageops::resize(&*cropped, MODEL_WIDTH, MODEL_HEIGHT, image::imageops::FilterType::Triangle);
  let mut data = Vec::with_capacity(MODEL_FRAME_BYTES);
  for pixel in resized.pixels() {
    data.extend_from_slice(&pixel.0[..3]);
  }

  Ok((
    recent.sequence,
    RgbFrame {
      width: MODEL_WIDTH,
      height: MODEL_HEIGHT,
      data,
    },
  ))
}

fn validate_rgba_frame(sequence: u64, frame: &RgbaFrame) -> Result<(), Box<dyn Error>> {
  let expected = usize::try_from(frame.width)
    .ok()
    .and_then(|width| usize::try_from(frame.height).ok().and_then(|height| width.checked_mul(height)))
    .and_then(|pixels| pixels.checked_mul(4))
    .ok_or_else(|| format!("RecentFrame sequence {sequence} dimensions overflow RGBA8 storage"))?;
  if frame.data.len() != expected {
    return Err(
      format!(
        "RecentFrame sequence {sequence} must contain tightly packed RGBA8, got {}x{} with {} bytes",
        frame.width,
        frame.height,
        frame.data.len()
      )
      .into(),
    );
  }
  Ok(())
}

fn infer(model: &OrtSession, frames: &[RgbFrame], held: &VecDeque<usize>, instruction: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
  let pixels = (MODEL_WIDTH * MODEL_HEIGHT) as usize;
  let mut frame_tensor = vec![0.0; FRAME_COUNT * 3 * pixels];
  let mean = [0.485, 0.456, 0.406];
  let std = [0.229, 0.224, 0.225];
  for (time, frame) in frames.iter().enumerate() {
    for (pixel_index, pixel) in frame.data.chunks_exact(3).enumerate() {
      for channel in 0..3 {
        let index = ((time * 3 + channel) * pixels) + pixel_index;
        frame_tensor[index] = (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
      }
    }
  }

  let mut held_tensor = vec![0.0; FRAME_COUNT * ACTIONS.len()];
  for (time, action) in held.iter().enumerate() {
    held_tensor[time * ACTIONS.len() + action] = 1.0;
  }

  let outputs = model.run_f32_many(vec![
    F32Tensor {
      name: "frames".to_owned(),
      shape: vec![
        1,
        FRAME_COUNT,
        3,
        MODEL_HEIGHT as usize,
        MODEL_WIDTH as usize,
      ],
      data: frame_tensor,
    },
    F32Tensor {
      name: "held".to_owned(),
      shape: vec![1, FRAME_COUNT, ACTIONS.len()],
      data: held_tensor,
    },
    F32Tensor {
      name: "instruction".to_owned(),
      shape: vec![1, TASKS.len() + TARGETS.len()],
      data: instruction.to_vec(),
    },
  ])?;
  let logits = outputs.first().ok_or("model returned no output")?;
  if logits.data.len() != ACTIONS.len() {
    return Err(format!("expected nine logits, got {}", logits.data.len()).into());
  }
  Ok(logits.data.clone())
}

fn action_key(action: usize) -> Option<&'static str> {
  match action {
    0 => Some("arrowup"),
    1 => Some("arrowdown"),
    2 => Some("arrowleft"),
    3 => Some("arrowright"),
    4 | 5 | 7 => Some("space"),
    6 => Some("q"),
    _ => None,
  }
}

fn save_evidence_frame(directory: &Path, name: &str, image: &RgbaImage) -> Result<(), image::ImageError> {
  image.save(directory.join(format!("evidence-{name}.png")))
}

fn merge_result<T>(primary: Result<T, Box<dyn Error>>, cleanup: Result<(), Box<dyn Error>>) -> Result<T, Box<dyn Error>> {
  match (primary, cleanup) {
    (Ok(value), Ok(())) => Ok(value),
    (Err(error), Ok(())) => Err(error),
    (Ok(_), Err(error)) => Err(error),
    (Err(primary), Err(cleanup)) => Err(format!("{primary}; cleanup also failed: {cleanup}").into()),
  }
}

fn write_subtitles(path: &Path, trace: &[String]) -> Result<(), std::io::Error> {
  let content = trace
    .iter()
    .enumerate()
    .map(|(index, line)| format!("{}\n00:00:{:02},000 --> 00:00:{:02},000\n{}\n", index + 1, index, index + 1, line))
    .collect::<Vec<_>>()
    .join("\n");
  std::fs::write(path, content)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn validates_instruction_pairs() {
    assert!(instruction("pickup", "iron").is_ok());
    assert!(instruction("attack", "monster").is_ok());
    assert!(instruction("enter", "mine").is_ok());
    assert!(instruction("attack", "iron").is_err());
  }

  #[test]
  fn maps_model_actions_to_game_keys() {
    assert_eq!(action_key(0), Some("arrowup"));
    assert_eq!(action_key(5), Some("space"));
    assert_eq!(action_key(NONE), None);
  }

  #[test]
  fn model_history_pads_the_front_with_the_first_available_frame() {
    let recent = [7, 8].into_iter().map(recent_frame).collect();
    let mut history = VecDeque::new();

    let (frames, first_sequence, last_sequence) = model_history(&mut history, recent, 0, 8).expect("build a complete Lower v0 history");

    assert_eq!(frames.len(), FRAME_COUNT);
    assert_eq!((first_sequence, last_sequence), (7, 8));
    assert!(frames[..FRAME_COUNT - 1].iter().all(|frame| frame.data[0] == 7));
    assert_eq!(frames.last().expect("padded history has a final frame").data[0], 8);
  }

  #[test]
  fn model_history_rejects_frames_outside_sequence_order() {
    let mut history = VecDeque::new();

    let error = model_history(&mut history, vec![recent_frame(2), recent_frame(1)], 0, 1).expect_err("out-of-order model frames must fail");

    assert_eq!(error.to_string(), "GetRecentFrames returned frames outside strict sequence order");
  }

  #[test]
  fn model_history_keeps_the_latest_ten_frames_across_responses() {
    let mut history = VecDeque::new();
    model_history(&mut history, (1..=8).map(recent_frame).collect(), 0, 8).expect("build initial history");

    let (frames, first_sequence, last_sequence) =
      model_history(&mut history, (9..=12).map(recent_frame).collect(), 8, 12).expect("extend history");

    assert_eq!(frames.len(), FRAME_COUNT);
    assert_eq!((first_sequence, last_sequence), (3, 12));
    assert_eq!(frames.first().expect("history has a first frame").data[0], 3);
    assert_eq!(frames.last().expect("history has a final frame").data[0], 12);
  }

  #[test]
  fn prepare_model_frame_excludes_the_title_bar() {
    let mut recent = recent_frame(7);
    let image = recent.capture.as_mut().and_then(|capture| capture.image.as_mut()).expect("test frame has an image");
    image.data[..TITLE_BAR_HEIGHT as usize * 4].fill(255);

    let (_, frame) = prepare_model_frame(recent).expect("prepare model frame");

    assert_eq!((frame.width, frame.height, frame.data.len()), (MODEL_WIDTH, MODEL_HEIGHT, MODEL_FRAME_BYTES));
    assert!(frame.data.chunks_exact(3).all(|pixel| pixel == [7, 0, 0]));
  }

  fn recent_frame(sequence: u64) -> frame_buffer_proto::RecentFrame {
    let rgba = [sequence as u8, 0, 0, 255];
    frame_buffer_proto::RecentFrame {
      sequence,
      capture: Some(frame_buffer_proto::CapturedFrame {
        image: Some(RgbaFrame {
          width: 1,
          height: TITLE_BAR_HEIGHT + 1,
          data: rgba.repeat((TITLE_BAR_HEIGHT + 1) as usize),
        }),
        ..Default::default()
      }),
    }
  }
}
