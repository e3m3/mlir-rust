// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;
use mlir::MlirOperation;

use std::cmp;
use std::fmt;

use crate::attributes;
use crate::interfaces;
use crate::ir;
use crate::traits;

use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use interfaces::Interface;
use attributes::NamedI64DenseArray;
use ir::Dialect;
use ir::Destroy;
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

#[derive(Clone)]
pub struct OperandSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct ResultSegmentSizes(MlirAttribute);

impl Destroy for dyn IROperation {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for dyn IROperation {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl cmp::PartialEq for dyn IROp {
    fn eq(&self, rhs: &Self) -> bool {
        self.to_string() == rhs.to_string()
    }
}

impl OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl From<MlirAttribute> for OperandSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        OperandSegmentSizes(attr)
    }
}

impl IRAttribute for OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for OperandSegmentSizes {
    fn get_name() -> &'static str {
        "operand_segment_sizes"
    }
}

impl NamedI64DenseArray for OperandSegmentSizes {}

impl ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl From<MlirAttribute> for ResultSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        ResultSegmentSizes(attr)
    }
}

impl IRAttribute for ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for ResultSegmentSizes {
    fn get_name() -> &'static str {
        "result_segment_sizes"
    }
}

impl NamedI64DenseArray for ResultSegmentSizes {}
