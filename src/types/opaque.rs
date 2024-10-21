// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirOpaqueTypeGet;
use mlir::mlirOpaqueTypeGetData;
use mlir::mlirOpaqueTypeGetDialectNamespace;
use mlir::mlirOpaqueTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Context;
use ir::StringRef;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Opaque(MlirType);

impl Opaque {
    pub fn new(context: &Context, namespace: &StringRef, data: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirOpaqueTypeGet(*context.get(), *namespace.get(), *data.get())))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_opaque() {
            eprint!("Cannot coerce type to opaque type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Opaque(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_opaque() {
            eprint!("Cannot coerce type to opaque type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_data(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetData(self.0)))
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueTypeGetDialectNamespace(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirOpaqueTypeGetTypeID()))
    }
}

impl IRType for Opaque {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }
}
