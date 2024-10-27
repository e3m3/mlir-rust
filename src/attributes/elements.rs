// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirElementsAttrGetNumElements;
use mlir::mlirElementsAttrGetValue;
use mlir::mlirElementsAttrIsValidIndex;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Shape;

#[derive(Clone)]
pub struct Elements(MlirAttribute);

impl Elements {
    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_elements() {
            eprint!("Cannot coerce attribute to elements attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Elements(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_value(&self, shape: &mut dyn Shape) -> Attribute {
        Attribute::from(do_unsafe!(mlirElementsAttrGetValue(
            self.0,
            shape.rank() as isize,
            shape.get().to_vec().as_mut_ptr(),
        )))
    }

    pub fn is_valid(&self, shape: &mut dyn Shape) -> bool {
        do_unsafe!(mlirElementsAttrIsValidIndex(
            self.0,
            shape.rank() as isize,
            shape.get().to_vec().as_mut_ptr(),
        ))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirElementsAttrGetNumElements(self.0)) as isize
    }
}

impl IRAttribute for Elements {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }
}
