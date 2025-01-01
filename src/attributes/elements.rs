// Copyright 2024-2025, Giordano Salvador
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

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Shape;

#[derive(Clone)]
pub struct Elements(MlirAttribute);

impl Elements {
    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_elements() {
            eprintln!("Cannot coerce attribute to elements attribute: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr)
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
            *self.get(),
            r,
            s.as_mut_ptr()
        )))
    }

    pub fn is_valid_value(&self, shape: &dyn Shape) -> bool {
        let r = shape.rank();
        let mut s: Vec<u64> = shape.to_vec().iter().map(|i| *i as u64).collect();
        do_unsafe!(mlirElementsAttrIsValidIndex(*self.get(), r, s.as_mut_ptr()))
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirElementsAttrGetNumElements(*self.get())) as isize
    }
}

impl From<MlirAttribute> for Elements {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Elements {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Elements {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Elements {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
