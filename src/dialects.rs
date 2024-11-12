// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;
use mlir::MlirOperation;

use std::cmp;
use std::fmt;
use std::hint::black_box;

use crate::attributes;
use crate::interfaces;
use crate::ir;
use crate::traits;

use attributes::dense_array::DenseArray;
use attributes::dense_array::Layout as DenseArrayLayout;
use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use interfaces::Interface;
use ir::Context;
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
    pub fn new(context: &Context, values: &[i64]) -> Self {
        Self::__from(*DenseArray::new_i64(context, values).get())
    }

    fn __from(attr: MlirAttribute) -> Self {
        OperandSegmentSizes(attr)
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Self::__from(attr);
        let _ = black_box(attr_.as_dense_array()); // Don't use the dense array, but check it.
        attr_
    }

    pub fn as_dense_array(&self) -> DenseArray {
        DenseArray::from(*self.get(), DenseArrayLayout::I64)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_name() -> &'static str {
        "operand_segment_sizes"
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
    fn get_name(&self) -> &'static str {
        OperandSegmentSizes::get_name()
    }
}

impl ResultSegmentSizes {
    pub fn new(context: &Context, values: &[i64]) -> Self {
        Self::__from(*DenseArray::new_i64(context, values).get())
    }

    fn __from(attr: MlirAttribute) -> Self {
        ResultSegmentSizes(attr)
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Self::__from(attr);
        let _ = black_box(attr_.as_dense_array()); // Don't use the dense array, but check it.
        attr_
    }

    pub fn as_dense_array(&self) -> DenseArray {
        DenseArray::from(*self.get(), DenseArrayLayout::I64)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_name() -> &'static str {
        "result_segment_sizes"
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
    fn get_name(&self) -> &'static str {
        ResultSegmentSizes::get_name()
    }
}
