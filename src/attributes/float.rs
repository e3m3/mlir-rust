// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirFloatAttrDoubleGet;
use mlir_sys::mlirFloatAttrDoubleGetChecked;
use mlir_sys::mlirFloatAttrGetValueDouble;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Location;
use types::IType;
use types::float::Float as FloatType;

#[derive(Clone)]
pub struct Float(MlirAttribute);

impl Float {
    pub fn new(t: &FloatType, value: f64) -> Self {
        let context = t.as_type().get_context();
        Self::from(do_unsafe!(mlirFloatAttrDoubleGet(
            *context.get(),
            *t.get(),
            value
        )))
    }

    pub fn new_checked(t: &FloatType, value: f64, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirFloatAttrDoubleGetChecked(
            *loc.get(),
            *t.get(),
            value
        )))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_float() {
            eprintln!("Cannot coerce attribute to float attribute: {}", attr);
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

    pub fn get_type_float(&self) -> FloatType {
        FloatType::from(*self.as_attribute().get_type().get())
    }

    pub fn get_value(&self) -> f64 {
        do_unsafe!(mlirFloatAttrGetValueDouble(*self.get()))
    }
}

impl From<MlirAttribute> for Float {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Float {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Float {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Float {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
