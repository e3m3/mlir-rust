// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirIntegerAttrGet;
use mlir::mlirIntegerAttrGetTypeID;
use mlir::mlirIntegerAttrGetValueInt;
use mlir::mlirIntegerAttrGetValueSInt;
use mlir::mlirIntegerAttrGetValueUInt;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::ir;
use crate::exit_code;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Type;
use ir::TypeID;

#[derive(Clone)]
pub struct Integer(MlirAttribute);

impl Integer {
    pub fn new(t: &Type, value: i64) -> Self {
        Self::from(do_unsafe!(mlirIntegerAttrGet(*t.get(), value)))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_integer() {
            eprint!("Cannot coerce attribute to integer attribute type: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Integer(attr)
    }

    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_int(&self) -> i64 {
        do_unsafe!(mlirIntegerAttrGetValueInt(self.0))
    }

    pub fn get_sint(&self) -> i64 {
        do_unsafe!(mlirIntegerAttrGetValueSInt(self.0))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerAttrGetTypeID()))
    }

    pub fn get_uint(&self) -> u64 {
        do_unsafe!(mlirIntegerAttrGetValueUInt(self.0))
    }
}

impl IRAttribute for Integer {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }
}
