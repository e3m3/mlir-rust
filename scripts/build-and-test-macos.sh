#!/bin/bash
# Copyright 2024-2025, Giordano Salvador
# SPDX-License-Identifier: BSD-3-Clause

ROOT_DIR="$(dirname $0)"
source "${ROOT_DIR}/setup-macos.sh"

cargo build --verbose ${build_mode}
cargo clippy --verbose ${build_mode}
cargo fmt --all -- --check
cargo fmt --all --manifest-path tests/lit-tests-rust/Cargo.toml -- --check
cargo test --verbose ${build_mode} -- --nocapture
