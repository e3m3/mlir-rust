// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirFunctionTypeGet;
use mlir::mlirFunctionTypeGetInput;
use mlir::mlirFunctionTypeGetNumInputs;
use mlir::mlirFunctionTypeGetNumResults;
use mlir::mlirFunctionTypeGetResult;
use mlir::mlirFunctionTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Function(MlirType);

impl Function {
    pub fn new(context: &Context, inputs: &[Type], results: &[Type]) -> Self {
        let i: Vec<MlirType> = inputs.iter().map(|t| *t.get()).collect();
        let r: Vec<MlirType> = results.iter().map(|t| *t.get()).collect();
        Self::from(do_unsafe!(mlirFunctionTypeGet(
            *context.get(),
            inputs.len() as isize,
            i.as_ptr(),
            results.len() as isize,
            r.as_ptr(),
        )))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_function() {
            eprint!("Cannot coerce type to function type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Function(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_function() {
            eprint!("Cannot coerce type to function type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_input(&self, i: isize) -> Type {
        if i >= self.num_inputs() || i < 0 {
            eprint!("Index '{}' out of bounds for function type input: ", i);
            self.as_type().dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirFunctionTypeGetInput(self.0, i)))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_result(&self, i: isize) -> Type {
        if i >= self.num_results() || i < 0 {
            eprint!("Index '{}' out of bounds for function type result: ", i);
            self.as_type().dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirFunctionTypeGetResult(self.0, i)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirFunctionTypeGetTypeID()))
    }

    pub fn num_inputs(&self) -> isize {
        do_unsafe!(mlirFunctionTypeGetNumInputs(self.0))
    }

    pub fn num_results(&self) -> isize {
        do_unsafe!(mlirFunctionTypeGetNumResults(self.0))
    }
}

impl IRType for Function {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
