// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::mlirIndexTypeGet;
use mlir_sys::mlirIndexTypeGetTypeID;
use mlir_sys::MlirType;

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
    const WIDTH: usize = 64;

    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIndexTypeGet(*context.get())))
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_index() {
            eprint!("Cannot coerce type to index type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Index(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    #[inline]
    pub const fn get_width(&self) -> usize {
        Self::WIDTH
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
