// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirUnrankedMemRefTypeGet;
use mlir::mlirUnrankedMemRefTypeGetChecked;
use mlir::mlirUnrankedMemRefTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Location;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct UnrankedMemRef(MlirType);

impl UnrankedMemRef {
    pub fn new(t: &Type, memory_space: &Attribute) -> Self {
        Self::from(do_unsafe!(mlirUnrankedMemRefTypeGet(*t.get(), *memory_space.get())))
    }

    pub fn new_checked(t: &Type, memory_space: &Attribute, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirUnrankedMemRefTypeGetChecked(*loc.get(), *t.get(), *memory_space.get())))
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unranked_mem_ref() {
            eprint!("Cannot coerce type to unranked mem ref type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        UnrankedMemRef(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirUnrankedMemRefTypeGetTypeID()))
    }
}

impl IRType for UnrankedMemRef {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
