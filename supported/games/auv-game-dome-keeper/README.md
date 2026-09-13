# AUV Dome Keeper

This package connects AUV to the frozen Lower v0 policy:

```text
Dome Keeper window
  -> built-in AUV recent frame buffer
  -> GetRecentFrames
  -> crop, resize, and RGB8 conversion
  -> Lower v0 ONNX
  -> action
  -> AUV input
```

The model has three inputs: ten frames, ten observed held-action slots, and one
task/target instruction. The bounded dome transition uses `enter/mine`.
The built-in Driver Runner captures native RGBA frames continuously at `target_fps`.
It skips missed capture ticks and does not queue capture work.
The controller removes the top 28 pixels from each captured frame.
It resizes the game content to 384 by 216 pixels and converts it to tight RGB8.
The controller retains the latest ten converted model frames across RPC calls.

One model frame contains 248,832 bytes.
Ten model frames contain 2,488,320 bytes, or approximately 2.37 MiB.
`GetRecentFrames` returns new retained frames in sequence order from oldest to newest.
If fewer than ten frames exist, the consumer prepends copies of the first frame.

Inference runs in the consumer process through ONNX Runtime.
Python is necessary only to export the training checkpoint.

The executable selects CoreML and has no provider option. It does not select CPU
when CoreML is unavailable. `auv-game-dome-keeper` supports only macOS.

## Run the controller

1. Build the AUV CLI and the controller.

```bash
mise x -- cargo build -p auv-cli -p auv-game-dome-keeper --bins
```

2. In a second terminal, start the decompiled game directly with the pinned
   Godot build. Disable all Mods so the AUV test observes the original game
   behavior.

```bash
cd ../airi-dome-keeper
mise x -- zsh -lc 'exec "$GODOT_BIN" --path "external/domekeeper-decompiled/$DOMEKEEPER_VERSION" --disable-mods --windowed --resolution 960x540'
```

3. From the AUV repository root, start the AUV daemon.

```bash
target/debug/auv serve
```

4. In a second terminal, run the controller through the AUV CLI.

```bash
PATH="$PWD/target/debug:$PATH" target/debug/auv game-dome-keeper \
  --model ../airi-dome-keeper/models/lower-v0/artifacts/frozen-20260913/model.onnx \
  --frames-dir ../airi-dome-keeper/recordings/dome-keeper-editor-onnx-live \
  --task enter \
  --target mine \
  --background-input \
  --target-fps 10 \
  --buffer-capacity 10
```

The AUV CLI supplies the inherited transport and the run context.
The daemon starts the built-in local Driver Runner for recent-frame capture.

On macOS, grant Accessibility and Screen Recording access to the controller process.
Grant Screen Recording access to the AUV Driver Runner.
The controller needs capture access only for the raw evidence screenshots.
The Driver Runner captures all model frames.

The controller resolves the exact visible `Dome Keeper (DEBUG)` window. It
saves `evidence-before.png` and `evidence-after.png` for review.
These raw evidence screenshots still include the 28-pixel title bar.
They do not enter the model frame buffer.

The local Godot test window is visible to ScreenCaptureKit but absent from its
Accessibility window tree. `--background-input` therefore delivers to the
captured window's process without AX focus. `--foreground-input` explicitly
activates the app, clicks the target window's content, confirms that the same
window is frontmost, restores the mouse position, and then uses global keyboard
input. It stops if that confirmation fails. Omit both flags when exact-window
foreground input works.

AUV sends each prediction as one complete key press. It cannot preserve
the policy's held-action state between decisions, so this proof still records
`none` as the observed held state.

On the current macOS development host, the Level consumed input from the old
model, while that model continued to predict `ui_down`. Before this crop, the
new model continuously predicted `none` during a live run. The executable proves
capture and native inference. It does not prove task completion or sustained
real-machine capture performance. Input success means event submission only.
Make sure that the game applies the input through a separate observation.
