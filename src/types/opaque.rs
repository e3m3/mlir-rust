// Copyright 2024, Giordano Salvador
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
use types::IRType;

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

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_opaque() {
            eprint!("Cannot coerce type to opaque type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Opaque(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_data(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetData(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetDialectNamespace(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirOpaqueTypeGetTypeID()))
    }
}

impl IRType for Opaque {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
