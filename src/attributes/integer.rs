// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirIntegerAttrGet;
use mlir_sys::mlirIntegerAttrGetTypeID;
use mlir_sys::mlirIntegerAttrGetValueInt;
use mlir_sys::mlirIntegerAttrGetValueSInt;
use mlir_sys::mlirIntegerAttrGetValueUInt;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::TypeID;
use types::IType;
use types::integer::Integer as IntegerType;

#[derive(Clone)]
pub struct Integer(MlirAttribute);

impl Integer {
    const WIDTH_INDEX: usize = 64;

    pub fn new(t: &IntegerType, value: i64) -> Self {
        Self::from(do_unsafe!(mlirIntegerAttrGet(*t.as_type().get(), value)))
    }

    pub fn new_index(context: &Context, value: i64) -> Self {
        let t = IntegerType::new_signless(context, Self::index_width()).as_type();
        Self::from(do_unsafe!(mlirIntegerAttrGet(*t.get(), value)))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_integer() {
            eprintln!(
                "Cannot coerce attribute to integer attribute type: {}",
                attr
            );
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
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
        do_unsafe!(mlirIntegerAttrGetValueUInt(*self.get()))
    }

    pub fn has_index_width(&self) -> bool {
        self.get_type_integer().get_width() == Self::index_width()
    }

    #[inline]
    pub const fn index_width() -> usize {
        Self::WIDTH_INDEX
    }
}

impl From<MlirAttribute> for Integer {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Integer {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Integer {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Integer {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
