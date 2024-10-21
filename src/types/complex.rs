// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirComplexTypeGet;
use mlir::mlirComplexTypeGetElementType;
use mlir::mlirComplexTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Complex(MlirType);

impl Complex {
    pub fn new(element: &Type) -> Self {
        Self::from(do_unsafe!(mlirComplexTypeGet(*element.get())))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_complex() {
            eprint!("Cannot coerce type to complex type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Complex(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_complex() {
            eprint!("Cannot coerce type to complex type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_element_type(&self) -> Type {
        Type::from(do_unsafe!(mlirComplexTypeGetElementType(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirComplexTypeGetTypeID()))
    }
}

impl IRType for Complex {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }
}
