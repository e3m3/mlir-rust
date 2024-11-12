// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use std::cmp;

use crate::ir;
use named::Named;
use ir::Attribute;
use ir::Identifier;
use ir::StringBacked;

pub mod array;
pub mod bool;
pub mod dense_array;
pub mod dense_elements;
pub mod dense_resource_elements;
pub mod dictionary;
pub mod elements;
pub mod float;
pub mod integer;
pub mod integer_set;
pub mod named;
pub mod opaque;
pub mod sparse_elements;
pub mod strided_layout;
pub mod string;
pub mod symbol_ref;
pub mod r#type;
pub mod unit;

pub trait IRAttribute {
    fn get(&self) -> &MlirAttribute;
    fn get_mut(&mut self) -> &mut MlirAttribute;

    fn as_attribute(&self) -> Attribute {
        Attribute::from(*self.get())
    }
}

pub trait IRAttributeNamed: IRAttribute {
    fn get_name(&self) -> &'static str;

    fn as_named_attribute(&self) -> Named {
        let attr = self.as_attribute();
        let context = attr.get_context();
        let name = StringBacked::from_string(&self.get_name().to_string());
        let id = Identifier::new(&context, &name.as_string_ref());
        Named::new(&id, &attr.as_attribute())
    }
}

impl cmp::PartialEq for dyn IRAttribute {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

impl cmp::PartialEq for dyn IRAttributeNamed {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}
