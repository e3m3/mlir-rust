// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirOpaqueAttrGet;
use mlir::mlirOpaqueAttrGetData;
use mlir::mlirOpaqueAttrGetDialectNamespace;
use mlir::mlirOpaqueAttrGetTypeID;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::StringRef;
use ir::Type;
use ir::TypeID;

#[derive(Clone)]
pub struct Opaque(MlirAttribute);

impl Opaque {
    pub fn new(context: &Context, namespace: &StringRef, data: &StringRef, t: &Type) -> Self {
        Self::from(do_unsafe!(mlirOpaqueAttrGet(
            *context.get(),
            *namespace.get(),
            data.len() as isize,
            data.as_ptr(),
            *t.get(),
        )))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_opaque() {
            eprint!("Cannot coerce attribute to opaque attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Opaque(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_data(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueAttrGetData(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueAttrGetDialectNamespace(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirOpaqueAttrGetTypeID()))
    }
}

impl IRAttribute for Opaque {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
