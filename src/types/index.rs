// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirIndexTypeGet;
use mlir_sys::mlirIndexTypeGetTypeID;

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
pub struct Index(MlirType);

impl Index {
    const WIDTH: usize = 64;

    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIndexTypeGet(*context.get())))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_index() {
            eprintln!("Cannot coerce type to index type: {}", t);
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

    #[inline]
    pub const fn get_width(&self) -> usize {
        Self::WIDTH
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIndexTypeGetTypeID()))
    }
}

impl From<MlirType> for Index {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Index {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Index {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Index {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
