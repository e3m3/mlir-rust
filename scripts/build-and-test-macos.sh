#!/bin/bash
# Copyright 2024, Giordano Salvador
# SPDX-License-Identifier: BSD-3-Clause

HOMEBREW_HOME=${HOMEBREW_HOME:=/opt/homebrew}
eval "$(${HOMEBREW_HOME}/bin/brew shellenv)"

LLVM_VER=19
PYTHON_VER=3.12

brew install python@${PYTHON_VER} llvm@${LLVM_VER} lit rustup

RUSTUP_CHANNEL=nightly-2024-12-10
RUSTUP_HOME=/root/.rustup

rustup toolchain install ${RUSTUP_CHANNEL}
rustup override set ${RUSTUP_CHANNEL}
rustup default ${RUSTUP_CHANNEL}

RUSTFLAGS=''
RUSTUP_TOOLCHAIN_PATH="$(rustup toolchain list | grep default | awk '{print $1}')"
RUSTUP_TOOLCHAIN="$(basename ${RUSTUP_TOOLCHAIN_PATH})"
RUST_SRC_PATH="${RUSTUP_TOOLCHAIN_PATH}/lib/rustlib/src/rust/src"
DYLD_LIBRARY_PATH="${RUSTUP_TOOLCHAIN_PATH}/lib:${DYLD_LIBRARY_PATH}"
PATH=${RUSTUP_HOME}/toolchains/${RUSTUP_TOOLCHAIN}/bin${PATH:+:$PATH}
LD_LIBRARY_PATH=${RUSTUP_HOME}/toolchains/${RUSTUP_TOOLCHAIN}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

LLVM_HOME=${HOMEBREW_HOME}/opt/llvm@${LLVM_VER}
PATH=${LLVM_HOME}/bin${PATH:+:$PATH}
LD_LIBRARY_PATH=${LLVM_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
C_INCLUDE_PATH=${LLVM_HOME}/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
CPLUS_INCLUDE_PATH=${LLVM_HOME}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}

PYTHON_VENV_PATH=/root/.python/venv
python3 -m venv ${PYTHON_VENV_PATH}
PATH=${PYTHON_VENV_PATH}/bin${PATH:+:$PATH}

export HOMEBREW_HOME
export LLVM_HOME
export PYTHON_VENV_PATH
export RUSTFLAGS
export RUST_SRC_PATH
export DYLD_LIBRARY_PATH

case ${BUILD_MODE} in
    debug)      build_mode= ;;
    release)    build_mode=--release ;;
    *)          echo "Error: BUILD_MODE=$BUILD_MODE" >2  &&  exit 1 ;;
esac

cargo build --verbose ${build_mode}
cargo clippy --verbose ${build_mode}
cargo fmt --all -- --check
cargo fmt --all --manifest-path tests/lit-tests-rust/Cargo.toml -- --check
cargo test --verbose ${build_mode} -- --nocapture
