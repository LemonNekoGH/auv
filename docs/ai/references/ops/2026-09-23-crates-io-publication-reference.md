# crates.io publication

## Publication set

The workspace contains 29 crates with `publish = true`. Cargo packages these
crates as one dependency set. The other 12 workspace crates use
`publish = false`.

Each internal path dependency has a registry version. Cargo removes the path
when it creates an archive. The registry version remains in the archive.

The `bumpp` hook updates `workspace.package.version` and each internal path
dependency version in the 29 publishable crates. It stops if a dependency
version is missing or differs from the previous workspace version. A `0.0.x`
version requirement accepts only that patch version.

## Source requirements

Clone the repository with recursive submodules. The `auv-media-macos` archive
includes the pinned `mediaremote-adapter` submodule.

The `auv-api-proto/proto` symlink points to the root `proto/` directory. Cargo
includes the required schemas as regular files in the crate archive.

On macOS, the build needs CMake and the Xcode Swift compiler. The Nix
development shell now provides CMake and exposes the system Swift compiler.

## Validation

The `cargo-package` CI job packages all 29 crates on macOS. Cargo compiles
each extracted archive against a temporary local registry. The normal Rust
matrix tests the workspace on Linux, macOS, and Windows.

The complete 29-crate package set passed on macOS arm64 on 2026-09-24. The run used
Cargo 1.96.0, CMake 4.1.2, and Apple Swift 6.2.1.

## Publication order

Publish the crates in these dependency levels. Wait until crates.io can resolve
one complete level before you publish the next level.

1. `auv-api-proto`, `auv-cli-common-macros`, `auv-cli-invoke-macros`,
   `auv-driver-common`, `auv-inference-common`, `auv-query-readiness`,
   `auv-tracing`
2. `auv-api-client`, `auv-cli-common`, `auv-driver-linux`,
   `auv-driver-overlay-common`, `auv-inference-ort`,
   `auv-inference-ultralytics`, `auv-media-macos`, `auv-tracing-otel`,
   `auv-view`
3. `auv-driver-overlay-macos`, `auv-driver-overlay-windows`,
   `auv-task-object-detection`
4. `auv-driver-overlay`
5. `auv-driver-macos`, `auv-driver-windows`
6. `auv-driver`
7. `auv-core`, `auv-scan`
8. `auv-api-server`, `auv-cli-invoke`
9. `auv-daemon`
10. `auv-cli`

Use `cargo publish --locked --package <name>` for each crate. Cargo cannot
upload several selected packages with one `cargo publish` command.

## Registry names

The core SDK package is named `auv-core`; this avoids the already registered
`auv` name. The crate remains at `crates/auv`, and workspace dependents retain
the `auv` Rust import through Cargo dependency aliases. Check ownership and
availability of every package name again immediately before publication. A
failure after lower-level uploads leaves a partial release on crates.io.
