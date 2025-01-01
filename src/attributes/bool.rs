// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirBoolAttrGet;
use mlir_sys::mlirBoolAttrGetValue;

use std::ffi::c_int;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;

#[derive(Clone)]
pub struct Bool(MlirAttribute);

impl Bool {
    pub fn new(context: &Context, value: bool) -> Self {
        Self::from(do_unsafe!(mlirBoolAttrGet(*context.get(), value as c_int)))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_bool() {
            eprintln!("Cannot coerce attribute to bool attribute: {}", attr);
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

    pub fn get_value(&self) -> bool {
        do_unsafe!(mlirBoolAttrGetValue(*self.get()))
    }
}

impl From<MlirAttribute> for Bool {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Bool {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Bool {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Bool {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
