---

#  Copyright

Copyright 2024, Giordano Salvador
SPDX-License-Identifier: BSD-3-Clause and CC-BY-SA-4.0


#   Original Design

Directory structure:

```
Cargo.toml
target/
    debug/ // == %O
tests/
    lit-tests-mlir/ // == %M
        lit.cfg
        test_1.mlir
            // RUN: @mlir-opt %s ...
            // CHECK: ...
            module {
                ...
            }
        ...
    lit-tests-rust/
        Cargo.toml
            [[bin]
            name = "test_1"
            path = "src/test_1.lit-rs"
            [[bin]]
            name = "test_2"
            path = "src/test_2.lit-rs"
            ...
        src/
            lit.cfg
            test_1.lit-rs
                // RUN: %O/test_1 2>&1 | mlir-opt | @filecheck %M/test_1.mlir
                // CHECK: ...
                pub fn main() {
                    ...
                }
            ...
    lit-tests.rs
        #[cfg(test)]
        mod tests {
            #[test]
            fn lit() {
                ...
            }
        }
```

##  Issues

1.  Monolithic rust test within `lit-tests.rs` which:

    *   Compiles rust lit test subproject binaries (one per test).

    *   Calls mlir lit test suite.

    *   Calls rust lit test suite (checked against mlir lit tests results).

1.  Running `cargo test` does not implicitly build rust lit test subproject
    (must be done within explicitly in `lit-test.rs`, or by user via subproject manifest path).

1.  Independent binaries for rust lit tests duplicate `mlir-rust` and `rust-src` library code,
    resulting in long compile times and large `target` directory size.
    Depending on the platform, large linking builds may crash test jobs [1].

1.  Each new rust lit test must be manually added as a binary in the subproject manifest.


#   New Design

Directory structure:

```
Cargo.toml
    [dependencies]
        ...
    [dev-dependencies]
        mlir-rust-tests = { path = "tests/lit-tests-rust" }
        ...
target/
    debug/ // == %O
tests/
    lit-tests-mlir/ // == %M
        lit.cfg
        test_1.mlir
            // RUN: @mlir-opt %s ...
            // CHECK: ...
            module {
                ...
            }
        ...
    lit-tests-rust/
        Cargo.toml
            [[bin]
            name = "run"
            path = "src/main.rs"
        build.rs
        src/
            lit.cfg
            common.rs
            default_test.rs
            main.rs
            test_1.lit-rs
                // RUN: %O/run test_1 2>&1 | mlir-opt | @filecheck %M/test_1.mlir
                // CHECK: ...
                pub fn test() -> TestResult {
                    ...
                }
            ...
    lit-tests.rs
        #[cfg(test)]
        mod tests {
            #[test]
            fn lit() {
                ...
            }
        }
```

##  Discussion

*   All tests are now compiled into one monolithic test binary called `run`.
    This binary is called by the `RUN` command in each lit test with the test name as the first argument.
    The `run` binary will check its registry of tests and callback the appropriate test function.

*   A `TestResult` value is now returned from each test, enabling custom error messages.

*   All tests (using the `lit-rs` file extension) are now automatically detected during the `build.rs`
    compilation phase.
    Each test's file name stem (e.g., `test_1` for `test_1.lit-rs`) will be registered with
    a standardized function callback with signature `pub fn test() -> TestResult`.

*   Launching `run` with an invalid test name will callback to the default `test_not_found`.

*   The current lit test structure retains the use of a single test result summary from lit
    for both MLIR and Rust based tests.

*   Test run-time (`cargo test`) has been reduced from several minutes to under 5 seconds.

*   Target directory (e.g., `target/debug`) space has been reduced from 20GB+ to under 1GB.

*   Full utilization of cores (up from 1) is restored for GitHub actions workflow build.


#   References

[1]:    https://github.com/e3m3/mlir-rust/issues/1

1.  `https://github.com/e3m3/mlir-rust/issues/1`
