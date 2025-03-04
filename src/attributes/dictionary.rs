// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirNamedAttribute;
use mlir_sys::mlirDictionaryAttrGet;
use mlir_sys::mlirDictionaryAttrGetElement;
use mlir_sys::mlirDictionaryAttrGetElementByName;
use mlir_sys::mlirDictionaryAttrGetNumElements;
use mlir_sys::mlirDictionaryAttrGetTypeID;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use attributes::named;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::StringRef;
use ir::TypeID;

#[derive(Clone)]
pub struct Dictionary(MlirAttribute);

impl Dictionary {
    pub fn new(context: &Context, elements: &[named::Named]) -> Self {
        let e: Vec<MlirNamedAttribute> = elements.iter().map(|a| *a.get()).collect();
        Self::from(do_unsafe!(mlirDictionaryAttrGet(
            *context.get(),
            e.len() as isize,
            e.as_ptr()
        )))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_dictionary() {
            eprintln!("Cannot coerce attribute to dictionary attribute: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_element(&self, i: isize) -> named::Named {
        named::Named::from(do_unsafe!(mlirDictionaryAttrGetElement(*self.get(), i)))
    }

    pub fn get_element_by_name(&self, name: &StringRef) -> Attribute {
        Attribute::from(do_unsafe!(mlirDictionaryAttrGetElementByName(
            *self.get(),
            *name.get()
        )))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirDictionaryAttrGetTypeID()))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirDictionaryAttrGetNumElements(*self.get()))
    }
}

impl From<MlirAttribute> for Dictionary {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Dictionary {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Dictionary {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Dictionary {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
