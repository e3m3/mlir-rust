// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::cmp;
use std::fmt;

use crate::interfaces;
use crate::ir;
use crate::traits;

use interfaces::Interface;
use ir::Dialect;
use ir::Operation;
use traits::Trait;

pub mod affine;
pub mod arith;
pub mod func;
//pub mod linalg;
//pub mod tensor;

///////////////////////////////
//  Traits
///////////////////////////////

/// Interface for dialect operations with trait and interface semantics.
pub trait DialectOperation {
    fn as_operation(&self) -> Operation;
    fn get_dialect(&self) -> Dialect;
    fn get_interfaces(&self) -> &'static [Interface];
    fn get_name(&self) -> &'static str;
    fn get_op(&self) -> &'static dyn DialectOp;
    fn get_traits(&self) -> &'static [Trait];
}

/// Interface for printable opcode.
pub trait DialectOp: fmt::Display {}

impl cmp::PartialEq for dyn DialectOp {
    fn eq(&self, rhs: &Self) -> bool {
        self.to_string() == rhs.to_string()
    }
}
