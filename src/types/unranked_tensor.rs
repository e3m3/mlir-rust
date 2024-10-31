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
        let t_ = Type::from(t);
        if !t_.is_unranked_tensor() {
            eprint!("Cannot coerce type to unranked tensor type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        UnrankedTensor(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unranked_tensor() {
            eprint!("Cannot coerce type to unranked tensor type: ");
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

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirUnrankedTensorTypeGetTypeID()))
    }
}

impl IRType for UnrankedTensor {
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
