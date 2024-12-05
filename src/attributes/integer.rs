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

use std::ffi::c_uint;

use crate::attributes;
use crate::do_unsafe;
use crate::ir;
use crate::exit_code;
use crate::types;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::TypeID;
use types::integer::Integer as IntegerType;
use types::IRType;

#[derive(Clone)]
pub struct Integer(MlirAttribute);

impl Integer {
    const WIDTH_INDEX: c_uint = 64;

    pub fn new(t: &IntegerType, value: i64) -> Self {
        Self::from(do_unsafe!(mlirIntegerAttrGet(*t.as_type().get(), value)))
    }

    pub fn new_index(context: &Context, value: i64) -> Self {
        let t = IntegerType::new_signless(context, Self::index_width()).as_type();
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

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_int(&self) -> i64 {
        do_unsafe!(mlirIntegerAttrGetValueInt(self.0))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_sint(&self) -> i64 {
        do_unsafe!(mlirIntegerAttrGetValueSInt(self.0))
    }

    pub fn get_type_integer(&self) -> IntegerType {
        IntegerType::from(*self.as_attribute().get_type().get())
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerAttrGetTypeID()))
    }

    pub fn get_uint(&self) -> u64 {
        do_unsafe!(mlirIntegerAttrGetValueUInt(self.0))
    }

    pub fn has_index_width(&self) -> bool {
        self.get_type_integer().get_width() == Self::index_width()
    }

    #[inline]
    pub const fn index_width() -> c_uint {
        Self::WIDTH_INDEX
    }
}

impl IRAttribute for Integer {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
