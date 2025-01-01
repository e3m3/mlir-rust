// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirNoneTypeGet;
use mlir_sys::mlirNoneTypeGetTypeID;

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
pub struct None(MlirType);

impl None {
    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirNoneTypeGet(*context.get())))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_none() {
            eprintln!("Cannot coerce type to none type: {}", t);
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

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirNoneTypeGetTypeID()))
    }
}

impl From<MlirType> for None {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for None {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for None {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for None {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
