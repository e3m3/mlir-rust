// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirFloatAttrDoubleGet;
use mlir::mlirFloatAttrDoubleGetChecked;
use mlir::mlirFloatAttrGetValueDouble;
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
use ir::Location;
use ir::Type;

#[derive(Clone)]
pub struct Float(MlirAttribute);

impl Float {
    pub fn new(context: &Context, t: &Type, value: f64) -> Self {
        Self::from(do_unsafe!(mlirFloatAttrDoubleGet(*context.get(), *t.get(), value)))
    }

    pub fn new_checked(t: &Type, value: f64, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirFloatAttrDoubleGetChecked(*loc.get(), *t.get(), value)))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_float() {
            eprint!("Cannot coerce attribute to float attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Float(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_value(&self) -> f64 {
        do_unsafe!(mlirFloatAttrGetValueDouble(self.0))
    }
}

impl IRAttribute for Float {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }
}
