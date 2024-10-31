// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirTypeEqual;
use mlir::MlirType;

use std::cmp;

use crate::do_unsafe;
use crate::ir;
use ir::Type;

pub mod complex;
pub mod float;
pub mod function;
pub mod integer;
pub mod mem_ref;
pub mod none;
pub mod opaque;
pub mod ranked_tensor;
pub mod shaped;
pub mod tuple;
pub mod vector;
pub mod unranked_mem_ref;
pub mod unranked_tensor;

pub trait IRType {
    fn as_type(&self) -> Type;
    fn get(&self) -> &MlirType;
    fn get_mut(&mut self) -> &mut MlirType;
}

impl dyn IRType {
    fn as_type(&self) -> Type {
        Type::from(*self.get())
    }
}

impl cmp::PartialEq for dyn IRType {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirTypeEqual(*self.get(), *rhs.get()))
    }
}
