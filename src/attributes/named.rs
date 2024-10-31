// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirNamedAttributeGet;
use mlir::MlirAttribute;
use mlir::MlirNamedAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::ir;

use attributes::IRAttribute;
use ir::Attribute;
use ir::Identifier;

#[derive(Clone)]
pub struct Named(MlirNamedAttribute);

impl Named {
    pub fn new(id: &Identifier, attr: &Attribute) -> Self {
        Named::from(do_unsafe!(mlirNamedAttributeGet(*id.get(), *attr.get())))
    }

    pub fn from(attr: MlirNamedAttribute) -> Self {
        Named(attr)
    }

    pub fn get(&self) -> &MlirNamedAttribute {
        &self.0
    }

    pub fn get_identifier(&self) -> Identifier {
        Identifier::from(self.0.name)
    }

    pub fn get_mut(&mut self) -> &mut MlirNamedAttribute {
        &mut self.0
    }
}

impl IRAttribute for Named {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0.attribute)
    }

    fn get(&self) -> &MlirAttribute {
        &self.0.attribute
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0.attribute
    }
}
