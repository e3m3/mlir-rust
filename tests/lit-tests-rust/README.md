#  Copyright

Copyright 2024, Giordano Salvador
SPDX-License-Identifier: BSD-3-Clause

Author/Maintainer:  Giordano Salvador <73959795+e3m3@users.noreply.github.com>


#  Description (Lit tests for MLIR Rust Bindings)
Lit tests for rust bindings wrapper over the [MLIR][1] [[1]] ([license][2] [[2]]) C API bindings
from the `mlir_sys` crate ([site][3] [[3]] and [license][4] [[4]]).


##  Prerequisites

*   rust-2021

*   mlir-rust crate library (as parent cargo project)

##  Setup

*   To run the full suite of tests, run the following command from the parent cargo project:
    
    ```shell
    cargo test -- --nocapture
    ```

*   To get the MLIR output for a specific test, run the binary for the given test after building:

    ```shell
    cargo build
    cargo run --bin <test-name>
    ```

    For example, to run the first test for floating point addition from the `arith` dialect:

    ```shell
    cargo run --bin arith_addf_1
    ```

    The list of available/enabled tests is contained in `Cargo.toml`.


#  References

[1]:    https://mlir.llvm.org/

[2]:    https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT

[3]:    https://crates.io/crates/mlir-sys

[4]:    https://github.com/femtomc/mlir-sys/blob/main/LICENSE

1.  `https://mlir.llvm.org/`

1.  `https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT`

1.  `https://crates.io/crates/mlir-sys`

1.  `https://github.com/femtomc/mlir-sys/blob/main/LICENSE`
