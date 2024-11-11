// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirIndexTypeGet;
use mlir::mlirIndexTypeGetTypeID;
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
pub struct Index(MlirType);

impl Index {
    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIndexTypeGet(*context.get())))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_index() {
            eprint!("Cannot coerce type to index type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Index(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_index() {
            eprint!("Cannot coerce type to index type: ");
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
        TypeID::from(do_unsafe!(mlirIndexTypeGetTypeID()))
    }
}

impl IRType for Index {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

