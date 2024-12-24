// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirOperation;

use std::cmp;
use std::fmt;

use crate::effects;
use crate::interfaces;
use crate::ir;
use crate::traits;

use effects::MemoryEffectList;
use interfaces::Interface;
use ir::Destroy;
use ir::Dialect;
use ir::Operation;
use traits::Trait;

pub mod affine;
pub mod arith;
//pub mod cf;
pub mod common;
pub mod func;
//pub mod gpu;
//pub mod index;
pub mod linalg;
//pub mod llvm;
//pub mod math;
pub mod memref;
//pub mod scf;
//pub mod spirv;
pub mod tensor;
pub mod ub;
pub mod vector;

///////////////////////////////
//  Traits
///////////////////////////////

/// Interface for printable opcode.
pub trait IOp: fmt::Display {
    fn get_name(&self) -> &'static str;
}

/// Interface for dialect operations with trait and interface semantics.
pub trait IOperation {
    fn get(&self) -> &MlirOperation;
    fn get_dialect(&self) -> Dialect;
    fn get_effects(&self) -> MemoryEffectList;
    fn get_interfaces(&self) -> &'static [Interface];
    fn get_mut(&mut self) -> &mut MlirOperation;
    fn get_name(&self) -> &'static str;
    fn get_op(&self) -> &'static dyn IOp;
    fn get_traits(&self) -> &'static [Trait];

    fn as_operation(&self) -> Operation {
        Operation::from(*self.get())
    }
}

impl cmp::PartialEq for dyn IOp {
    fn eq(&self, rhs: &Self) -> bool {
        self.to_string() == rhs.to_string()
    }
}

impl Destroy for dyn IOperation {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for dyn IOperation {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}
