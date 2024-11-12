// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirUnrankedTensorTypeGet;
use mlir::mlirUnrankedTensorTypeGetChecked;
use mlir::mlirUnrankedTensorTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Location;
use ir::Type;
use ir::TypeID;
use types::IRType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct UnrankedTensor(MlirType);

impl UnrankedTensor {
    pub fn new(t: &Type) -> Self {
        Self::from(do_unsafe!(mlirUnrankedTensorTypeGet(*t.get())))
    }

    pub fn new_checked(t: &Type, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirUnrankedTensorTypeGetChecked(*loc.get(), *t.get())))
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unranked_tensor() {
            eprint!("Cannot coerce type to unranked tensor type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        UnrankedTensor(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(self.0)
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirUnrankedTensorTypeGetTypeID()))
    }
}

impl IRType for UnrankedTensor {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
