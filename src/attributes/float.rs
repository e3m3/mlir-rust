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
use crate::types;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Location;
use types::float::Float as FloatType;
use types::IRType;

#[derive(Clone)]
pub struct Float(MlirAttribute);

impl Float {
    pub fn new(t: &FloatType, value: f64) -> Self {
        if !t.is_f64() {
            eprintln!("Only double types are supported for float attributes");
            exit(ExitCode::IRError);
        }
        let context = t.as_type().get_context();
        Self::from(do_unsafe!(mlirFloatAttrDoubleGet(*context.get(), *t.get(), value)))
    }

    pub fn new_checked(t: &FloatType, value: f64, loc: &Location) -> Self {
        if !t.is_f64() {
            eprintln!("Only double types are supported for float attributes");
            exit(ExitCode::IRError);
        }
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

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_float(&self) -> FloatType {
        FloatType::from(*self.as_attribute().get_type().get())
    }

    pub fn get_value(&self) -> f64 {
        do_unsafe!(mlirFloatAttrGetValueDouble(self.0))
    }
}

impl IRAttribute for Float {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
