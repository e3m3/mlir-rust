// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirTupleTypeGet;
use mlir::mlirTupleTypeGetNumTypes;
use mlir::mlirTupleTypeGetType;
use mlir::mlirTupleTypeGetTypeID;
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
pub struct Tuple(MlirType);

impl Tuple {
    pub fn new(context: &Context, elements: &[Type]) -> Self {
        let e: Vec<MlirType> = elements.iter().map(|t| *t.get()).collect();
        Self::from(do_unsafe!(mlirTupleTypeGet(*context.get(), elements.len() as isize, e.as_ptr())))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_tuple() {
            eprint!("Cannot coerce type to tuple type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Tuple(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_tuple() {
            eprint!("Cannot coerce type to tuple type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type(&self, i: isize) -> Type {
        if i >= self.num_types() || i < 0 {
            eprint!("Index '{}' out of bounds for tuple type: ", i);
            self.as_type().dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Type::from(do_unsafe!(mlirTupleTypeGetType(self.0, i)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirTupleTypeGetTypeID()))
    }

    pub fn num_types(&self) -> isize {
        do_unsafe!(mlirTupleTypeGetNumTypes(self.0))
    }
}

impl IRType for Tuple {
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
