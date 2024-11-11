// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirArrayAttrGet;
use mlir::mlirArrayAttrGetElement;
use mlir::mlirArrayAttrGetNumElements;
use mlir::mlirArrayAttrGetTypeID;
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
use ir::TypeID;

#[derive(Clone)]
pub struct Array(MlirAttribute);

impl Array {
    pub fn new(context: &Context, elements: &[Attribute]) -> Self {
        let e: Vec<MlirAttribute> = elements.iter().map(|a| *a.get()).collect();
        Self::from(do_unsafe!(mlirArrayAttrGet(*context.get(), e.len() as isize, e.as_ptr())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_array() {
            eprint!("Cannot coerce attribute to array attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Array(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_element(&self, i: isize) -> Attribute {
        Attribute::from(do_unsafe!(mlirArrayAttrGetElement(self.0, i)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirArrayAttrGetTypeID()))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirArrayAttrGetNumElements(self.0))
    }
}

impl IRAttribute for Array {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
