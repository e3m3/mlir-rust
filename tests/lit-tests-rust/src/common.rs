// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]
//  ^Attribute separator

extern crate mlir;

use crate::default_test::DEFAULT_TEST_NAME;

use mlir::dialects::common::SymbolVisibilityKind;
use mlir::dialects::func::Arguments;
use mlir::dialects::func::Func;
use mlir::dialects::func::Results;
use mlir::ir::Context;
use mlir::ir::Module;
use mlir::ir::Pass;
use mlir::ir::Registry;
use mlir::ir::StringBacked;
use mlir::ir::Type;
use mlir::types::function::Function as FunctionType;

use std::collections::HashMap;
use std::process;

#[repr(i32)]
pub enum ExitCode {
    Ok = 0,
    ArgError = 100,
    TestError = 200,
}

pub type TestCallback = fn() -> TestResult;
pub type TestResult = Result<(), String>;

pub struct TestRegistry {
    map: HashMap<String, TestCallback>,
}

pub fn exit(code: ExitCode) -> ! {
    process::exit(code as i32);
}

pub fn get_empty_test_fn(context: &Context, inputs: &[Type], results: &[Type]) -> Func {
    get_fn(
        context,
        "test",
        inputs,
        results,
        SymbolVisibilityKind::None,
        None,
        None,
    )
}

pub fn get_fn(
    context: &Context,
    name: &str,
    inputs: &[Type],
    results: &[Type],
    visibility: SymbolVisibilityKind,
    input_attrs: Option<&Arguments>,
    result_attrs: Option<&Results>,
) -> Func {
    let name = StringBacked::from(name);
    let loc = context.get_unknown_location();
    let t_f = FunctionType::new(context, inputs, results);
    Func::new(
        &t_f,
        &name.as_string_ref(),
        visibility,
        input_attrs,
        result_attrs,
        &loc,
    )
}

pub fn get_module(registry: &Registry) -> Module {
    Pass::register_all_passes();
    let context = Context::from_registry(registry);
    Module::new(&context.get_unknown_location())
}

pub fn get_private_fn(
    context: &Context,
    name: &str,
    inputs: &[Type],
    results: &[Type],
    input_attrs: Option<&Arguments>,
    result_attrs: Option<&Results>,
) -> Func {
    get_fn(
        context,
        name,
        inputs,
        results,
        SymbolVisibilityKind::Private,
        input_attrs,
        result_attrs,
    )
}

pub fn get_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_arith();
    registry.register_func();
    registry.register_gpu();
    registry.register_linalg();
    registry.register_llvm();
    registry.register_memref();
    registry.register_shape();
    registry.register_spirv();
    registry.register_tensor();
    registry.register_vector();
    registry
}

impl TestRegistry {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn add_test(&mut self, test_name: &str, test_callback: TestCallback) -> () {
        if self
            .map
            .insert(test_name.to_string(), test_callback)
            .is_some()
        {
            eprintln!("Attempted to register test '{}' more than once", test_name);
            exit(ExitCode::TestError);
        }
    }

    pub fn get_test(&self, test_name: &str) -> TestCallback {
        match self.map.get(test_name) {
            Some(cb) => *cb,
            None => self.get_test(DEFAULT_TEST_NAME),
        }
    }
}

impl Default for TestRegistry {
    fn default() -> Self {
        Self::new()
    }
}
