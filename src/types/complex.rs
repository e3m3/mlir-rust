// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirComplexTypeGet;
use mlir_sys::mlirComplexTypeGetElementType;
use mlir_sys::mlirComplexTypeGetTypeID;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Type;
use ir::TypeID;
use types::IType;

#[derive(Clone)]
pub struct Complex(MlirType);

impl Complex {
    pub fn new(element: &Type) -> Self {
        Self::from(do_unsafe!(mlirComplexTypeGet(*element.get())))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_complex() {
            eprintln!("Cannot coerce type to complex type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_element_type(&self) -> Type {
        Type::from(do_unsafe!(mlirComplexTypeGetElementType(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirComplexTypeGetTypeID()))
    }
}

impl From<MlirType> for Complex {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Complex {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Complex {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Complex {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
