// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirAttributeEqual;
use mlir::MlirAttribute;

use std::cmp;

use crate::do_unsafe;
use crate::ir;
use ir::Attribute;

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
    fn as_attribute(&self) -> Attribute;
    fn get(&self) -> &MlirAttribute;
    fn get_mut(&mut self) -> &mut MlirAttribute;
}

impl dyn IRAttribute {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(*self.get())
    }
}

impl cmp::PartialEq for dyn IRAttribute {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAttributeEqual(*self.get(), *rhs.get()))
    }
}
