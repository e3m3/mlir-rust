// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirOpaqueTypeGet;
use mlir_sys::mlirOpaqueTypeGetData;
use mlir_sys::mlirOpaqueTypeGetDialectNamespace;
use mlir_sys::mlirOpaqueTypeGetTypeID;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Context;
use ir::StringRef;
use ir::Type;
use ir::TypeID;
use types::IType;

#[derive(Clone)]
pub struct Opaque(MlirType);

impl Opaque {
    pub fn new(context: &Context, namespace: &StringRef, data: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirOpaqueTypeGet(
            *context.get(),
            *namespace.get(),
            *data.get()
        )))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_opaque() {
            eprintln!("Cannot coerce type to opaque type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_data(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetData(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetDialectNamespace(*self.get())))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirOpaqueTypeGetTypeID()))
    }
}

impl From<MlirType> for Opaque {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Opaque {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Opaque {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Opaque {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
