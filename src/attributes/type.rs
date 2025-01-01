// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirTypeAttrGet;
use mlir_sys::mlirTypeAttrGetTypeID;
use mlir_sys::mlirTypeAttrGetValue;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::TypeID;

#[derive(Clone)]
pub struct Type(MlirAttribute);

impl Type {
    pub fn new(t: &ir::Type) -> Self {
        Self::from(do_unsafe!(mlirTypeAttrGet(*t.get())))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_type() {
            eprintln!("Cannot coerce attribute to type attribute: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type(&self) -> ir::Type {
        ir::Type::from(do_unsafe!(mlirTypeAttrGetValue(*self.get())))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirTypeAttrGetTypeID()))
    }
}

impl From<MlirAttribute> for Type {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Type {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Type {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Type {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
