// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirOperation;

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
pub trait IROperation {
    fn get(&self) -> &MlirOperation;
    fn get_dialect(&self) -> Dialect;
    fn get_interfaces(&self) -> &'static [Interface];
    fn get_mut(&mut self) -> &mut MlirOperation;
    fn get_name(&self) -> &'static str;
    fn get_op(&self) -> &'static dyn IROp;
    fn get_traits(&self) -> &'static [Trait];

    fn as_operation(&self) -> Operation {
        Operation::from(*self.get())
    }
}

/// Interface for printable opcode.
pub trait IROp: fmt::Display {
    fn get_name(&self) -> &'static str;
}

impl cmp::PartialEq for dyn IROp {
    fn eq(&self, rhs: &Self) -> bool {
        self.to_string() == rhs.to_string()
    }
}
