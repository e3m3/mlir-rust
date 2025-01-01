// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirTupleTypeGet;
use mlir_sys::mlirTupleTypeGetNumTypes;
use mlir_sys::mlirTupleTypeGetType;
use mlir_sys::mlirTupleTypeGetTypeID;

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
pub struct Tuple(MlirType);

impl Tuple {
    pub fn new(context: &Context, elements: &[Type]) -> Self {
        let e: Vec<MlirType> = elements.iter().map(|t| *t.get()).collect();
        Self::from(do_unsafe!(mlirTupleTypeGet(
            *context.get(),
            elements.len() as isize,
            e.as_ptr()
        )))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_tuple() {
            eprintln!("Cannot coerce type to tuple type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type(&self, i: isize) -> Type {
        if i >= self.num_types() || i < 0 {
            eprintln!(
                "Index '{}' out of bounds for tuple type: {}",
                i,
                self.as_type()
            );
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirTupleTypeGetType(*self.get(), i)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirTupleTypeGetTypeID()))
    }

    pub fn num_types(&self) -> isize {
        do_unsafe!(mlirTupleTypeGetNumTypes(*self.get()))
    }
}

impl From<MlirType> for Tuple {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Tuple {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Tuple {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Tuple {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
