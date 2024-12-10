// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirTypeEqual;
use mlir::MlirType;

use std::cmp;

use crate::do_unsafe;
use crate::ir;

use ir::Context;
use ir::Type;

use float::Float as FloatType;
use index::Index;
use integer::Integer as IntegerType;
use shaped::Shaped;

pub mod complex;
pub mod float;
pub mod function;
pub mod index;
pub mod integer;
pub mod memref;
pub mod none;
pub mod opaque;
pub mod ranked_tensor;
pub mod shaped;
pub mod tuple;
pub mod unit;
pub mod unranked_memref;
pub mod unranked_tensor;
pub mod vector;

pub trait GetWidth: IRType {
    fn get_width(&self) -> Option<usize> {
        if self.as_type().is_index() {
            Some(Index::from(*self.get()).get_width())
        } else if self.as_type().is_integer() {
            Some(IntegerType::from(*self.get()).get_width())
        } else if self.as_type().is_float() {
            Some(FloatType::from(*self.get()).get_width())
        } else if self.as_type().is_shaped() {
            Shaped::from(*self.get()).get_element_type().get_width()
        } else {
            None
        }
    }
}

pub trait IRType {
    fn get(&self) -> &MlirType;
    fn get_mut(&mut self) -> &mut MlirType;

    fn as_type(&self) -> Type {
        Type::from(*self.get())
    }

    fn get_context(&self) -> Context {
        self.as_type().get_context()
    }
}

pub trait IsPromotableTo<T> {
    fn is_promotable_to(&self, other: &T) -> bool;
}

impl GetWidth for dyn IRType {}

impl cmp::PartialEq for dyn IRType {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirTypeEqual(*self.get(), *rhs.get()))
    }
}

impl <T: IRType> IsPromotableTo<T> for dyn IRType {
    fn is_promotable_to(&self, other: &T) -> bool {
        let t = self.as_type();
        let t_other = other.as_type();
        if t.is_integer() && t_other.is_integer() {
            let t_int = IntegerType::from(*t.get());
            let t_int_other = IntegerType::from(*t_other.get());
            t_int.is_promotable_to(&t_int_other)
        } else if t.is_float() && t_other.is_float() {
            let t_float = FloatType::from(*t.get());
            let t_float_other = FloatType::from(*t_other.get());
            t_float.is_promotable_to(&t_float_other)
        } else {
            false
        }
    }
}
