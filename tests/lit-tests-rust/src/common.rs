// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir;

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

pub fn get_empty_test_fn(context: &Context, inputs: &[Type], results: &[Type]) -> Func {
    let name = StringBacked::from_string(&"test".to_string());
    let loc = context.get_unknown_location();
    let t_f = FunctionType::new(context, inputs, results);
    Func::new(
        &t_f,
        &name.as_string_ref(),
        SymbolVisibilityKind::None,
        None,
        None,
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
    let name = StringBacked::from_string(&name.to_string());
    let loc = context.get_unknown_location();
    let t_f = FunctionType::new(context, inputs, results);
    Func::new(
        &t_f,
        &name.as_string_ref(),
        SymbolVisibilityKind::Private,
        input_attrs,
        result_attrs,
        &loc,
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
