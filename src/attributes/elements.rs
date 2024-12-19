// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirElementsAttrGetNumElements;
use mlir_sys::mlirElementsAttrGetValue;
use mlir_sys::mlirElementsAttrIsValidIndex;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::ExitCode;
use exit_code::exit;
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

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_value(&self, shape: &dyn Shape) -> Attribute {
        let r = shape.rank();
        let mut s: Vec<u64> = shape.to_vec().iter().map(|i| *i as u64).collect();
        Attribute::from(do_unsafe!(mlirElementsAttrGetValue(
            self.0,
            r,
            s.as_mut_ptr()
        )))
    }

    pub fn is_valid_value(&self, shape: &dyn Shape) -> bool {
        let r = shape.rank();
        let mut s: Vec<u64> = shape.to_vec().iter().map(|i| *i as u64).collect();
        do_unsafe!(mlirElementsAttrIsValidIndex(self.0, r, s.as_mut_ptr()))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirElementsAttrGetNumElements(self.0)) as isize
    }
}

impl IRAttribute for Elements {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
