// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirArrayAttrGet;
use mlir_sys::mlirArrayAttrGetElement;
use mlir_sys::mlirArrayAttrGetNumElements;
use mlir_sys::mlirArrayAttrGetTypeID;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::TypeID;

#[derive(Clone)]
pub struct Array(MlirAttribute);

impl Array {
    pub fn new(context: &Context, elements: &[Attribute]) -> Self {
        let e: Vec<MlirAttribute> = elements.iter().map(|a| *a.get()).collect();
        Self::from(do_unsafe!(mlirArrayAttrGet(
            *context.get(),
            e.len() as isize,
            e.as_ptr()
        )))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_array() {
            eprintln!("Cannot coerce attribute to array attribute: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_element(&self, i: isize) -> Attribute {
        Attribute::from(do_unsafe!(mlirArrayAttrGetElement(*self.get(), i)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirArrayAttrGetTypeID()))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirArrayAttrGetNumElements(*self.get()))
    }
}

impl From<MlirAttribute> for Array {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Array {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Array {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Array {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
