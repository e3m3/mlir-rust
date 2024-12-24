#  Copyright

Copyright 2024, Giordano Salvador
SPDX-License-Identifier: BSD-3-Clause

Author/Maintainer:  Giordano Salvador <73959795+e3m3@users.noreply.github.com>


#  Description (MLIR Rust Bindings)

[![Fedora 41](https://github.com/e3m3/mlir-rust/actions/workflows/fedora-41.yaml/badge.svg?event=workflow_dispatch)](https://github.com/e3m3/mlir-rust/actions/workflows/fedora-40.yaml)

[![MacOS 14](https://github.com/e3m3/mlir-rust/actions/workflows/macos-14.yaml/badge.svg?event=workflow_dispatch)](https://github.com/e3m3/mlir-rust/actions/workflows/macos-14.yaml)

Rust bindings wrapper over the [MLIR][1] [[1]] ([license][2] [[2]]) C API bindings from the 
`mlir_sys` crate ([site][3] [[3]] and [license][4] [[4]]).


##  Prerequisites

*   rust-2021

*   llvm-19, clang-19, and mlir-sys (or llvm version matching mlir-sys)

*   libxml2 and libz headers (for testing)

*   num_cpus (for testing)

*   python3-lit, FileCheck (for testing)

    *   By default, `tests/lit-tests.rs` will search for the lit executable in
        `$PYTHON_VENV_PATH/bin` (if it exists) or the system's `/usr/bin`.

*   [docker|podman] (for testing/containerization)

    *   A [Fedora][5] [[5]] image can be built using `containers/Containerfile.fedora*`.

##  Setup

*   Native build:
    
    ```shell
    cargo build
    ```

*   Build Rust lit tests:

    ```shell
    cargo build --manifest-path tests/lit-tests-rust/Cargo.toml
    ```

*   Run Rust lit test (e.g., `tests/lit-tests-rust/src/<test-name>.lit-rs`:

    ```shell
    cargo run --manifest-path tests/lit-tests-rust/Cargo.toml -- <test-name>
    ```

*   Build and run lit test suite (MLIR + Rust):

    ```shell
    cargo test -- --nocapture
    ```

*   Container build and test [podman][6] [[6]]:

    ```shell
    podman build -t calcc -f container/Containerfile .
    ```

*   Container build and test [docker][7] [[7]]:

    ```shell
    docker build -t calcc -f container/Dockerfile .
    ```

*   If `make` is installed, you can build the image by running:

    ```shell
    make
    ```


#  References

[1]:    https://mlir.llvm.org/

[2]:    https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT

[3]:    https://crates.io/crates/mlir-sys

[4]:    https://github.com/mlir-rs/mlir-sys/blob/main/LICENSE

[5]:    https://fedoraproject.org/

[6]:    https://podman.io/

[7]:    https://www.docker.com/

1.  `https://mlir.llvm.org/`

1.  `https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT`

1.  `https://crates.io/crates/mlir-sys`

1.  `https://github.com/mlir-rs/mlir-sys/blob/main/LICENSE`

1.  `https://fedoraproject.org/`

1.  `https://podman.io/`

1.  `https://www.docker.com/`
