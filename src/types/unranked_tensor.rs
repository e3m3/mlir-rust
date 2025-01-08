// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirUnrankedTensorTypeGet;
use mlir_sys::mlirUnrankedTensorTypeGetChecked;
use mlir_sys::mlirUnrankedTensorTypeGetTypeID;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Location;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::shaped::NewElementType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct UnrankedTensor(MlirType);

impl UnrankedTensor {
    pub fn new(t: &Type) -> Self {
        Self::from(do_unsafe!(mlirUnrankedTensorTypeGet(*t.get())))
    }

    pub fn new_checked(t: &Type, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirUnrankedTensorTypeGetChecked(
            *loc.get(),
            *t.get()
        )))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unranked_tensor() {
            eprint!("Cannot coerce type to unranked tensor type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(*self.get())
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

impl From<MlirType> for UnrankedTensor {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for UnrankedTensor {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for UnrankedTensor {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for UnrankedTensor {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl NewElementType for UnrankedTensor {
    fn new_element_type(_other: &Self, t: &Type) -> Self {
        Self::new(t)
    }
}
