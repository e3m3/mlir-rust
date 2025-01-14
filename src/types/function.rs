// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirFunctionTypeGet;
use mlir_sys::mlirFunctionTypeGetInput;
use mlir_sys::mlirFunctionTypeGetNumInputs;
use mlir_sys::mlirFunctionTypeGetNumResults;
use mlir_sys::mlirFunctionTypeGetResult;
use mlir_sys::mlirFunctionTypeGetTypeID;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IType;

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

    pub fn from_type(t: &Type) -> Self {
        if !t.is_function() {
            eprintln!("Cannot coerce type to function type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_input(&self, i: isize) -> Type {
        if i >= self.num_inputs() || i < 0 {
            eprintln!(
                "Index '{}' out of bounds for function type input: {}",
                i,
                self.as_type()
            );
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirFunctionTypeGetInput(*self.get(), i)))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_result(&self, i: isize) -> Type {
        if i >= self.num_results() || i < 0 {
            eprintln!(
                "Index '{}' out of bounds for function type result: {}",
                i,
                self.as_type()
            );
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirFunctionTypeGetResult(*self.get(), i)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirFunctionTypeGetTypeID()))
    }

    pub fn num_inputs(&self) -> isize {
        do_unsafe!(mlirFunctionTypeGetNumInputs(*self.get()))
    }

    pub fn num_results(&self) -> isize {
        do_unsafe!(mlirFunctionTypeGetNumResults(*self.get()))
    }
}

impl From<MlirType> for Function {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Function {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Function {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Function {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
