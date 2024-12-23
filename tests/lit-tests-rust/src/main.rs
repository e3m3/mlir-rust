// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::env;

mod default_test;
use crate::default_test::DEFAULT_TEST_NAME;

mod common;
use crate::common::ExitCode;
use crate::common::TestRegistry;
use crate::common::exit;

include!("../../../target/debug/lit.rs");

pub fn main() -> ! {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!(
            "Expected <test-name> as argument to '{}' test binary",
            args.first().cloned().unwrap_or_default(),
        );
        exit(ExitCode::ArgError);
    }
    let test_name = args
        .last()
        .cloned()
        .unwrap_or(DEFAULT_TEST_NAME.to_string());
    let mut test_registry = TestRegistry::new();
    register_test_callbacks(&mut test_registry);
    match test_registry.get_test(&test_name)() {
        Ok(()) => exit(ExitCode::Ok),
        Err(msg) => {
            eprintln!("{}", msg);
            exit(ExitCode::TestError);
        }
    }
}
