// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::mlirNamedAttributeGet;
use mlir_sys::MlirAttribute;
use mlir_sys::MlirNamedAttribute;

use std::cmp;

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
    fn get(&self) -> &MlirAttribute {
        &self.0.attribute
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0.attribute
    }
}

impl cmp::PartialEq for Named {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute() && self.get_identifier() == rhs.get_identifier()
    }
}
