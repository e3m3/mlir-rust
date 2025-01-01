// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirOpaqueAttrGet;
use mlir_sys::mlirOpaqueAttrGetData;
use mlir_sys::mlirOpaqueAttrGetDialectNamespace;
use mlir_sys::mlirOpaqueAttrGetTypeID;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::StringRef;
use ir::Type;
use ir::TypeID;

#[derive(Clone)]
pub struct Opaque(MlirAttribute);

impl Opaque {
    pub fn new(t: &Type, namespace: &StringRef, data: &StringRef) -> Self {
        let context = t.get_context();
        Self::from(do_unsafe!(mlirOpaqueAttrGet(
            *context.get(),
            *namespace.get(),
            data.len() as isize,
            data.as_ptr(),
            *t.get(),
        )))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_opaque() {
            eprintln!("Cannot coerce attribute to opaque attribute: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_data(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueAttrGetData(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirOpaqueAttrGetDialectNamespace(*self.get())))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirOpaqueAttrGetTypeID()))
    }
}

impl From<MlirAttribute> for Opaque {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Opaque {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Opaque {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Opaque {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
