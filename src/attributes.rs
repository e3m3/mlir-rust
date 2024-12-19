// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;

use std::cmp;

use crate::ir;

use ir::Attribute;
use ir::Context;
use ir::Identifier;
use ir::StringBacked;
use ir::Type;

use named::Named;

pub mod array;
pub mod bool;
pub mod dense_array;
pub mod dense_elements;
pub mod dense_resource_elements;
pub mod dictionary;
pub mod elements;
pub mod float;
pub mod index;
pub mod integer;
pub mod integer_set;
pub mod named;
pub mod opaque;
pub mod sparse_elements;
pub mod specialized;
pub mod strided_layout;
pub mod string;
pub mod symbol_ref;
pub mod r#type;
pub mod unit;

///////////////////////////////
//  Generic Traits
///////////////////////////////

pub trait IRAttribute {
    fn get(&self) -> &MlirAttribute;
    fn get_mut(&mut self) -> &mut MlirAttribute;

    fn as_attribute(&self) -> Attribute {
        Attribute::from(*self.get())
    }

    fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    fn get_type(&self) -> Type {
        self.as_attribute().get_type()
    }
}

pub trait IRAttributeNamed: IRAttribute {
    fn get_name() -> &'static str
    where
        Self: Sized;

    fn as_named_attribute(&self) -> Named
    where
        Self: Sized,
    {
        let attr = self.as_attribute();
        let context = attr.get_context();
        let name = StringBacked::from(Self::get_name());
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
