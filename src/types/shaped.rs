// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirShapedTypeGetDimSize;
use mlir::mlirShapedTypeGetDynamicSize;
use mlir::mlirShapedTypeGetDynamicStrideOrOffset;
use mlir::mlirShapedTypeGetElementType;
use mlir::mlirShapedTypeGetRank;
use mlir::mlirShapedTypeHasRank;
use mlir::mlirShapedTypeHasStaticShape;
use mlir::mlirShapedTypeIsDynamicDim;
use mlir::mlirShapedTypeIsDynamicSize;
use mlir::mlirShapedTypeIsDynamicStrideOrOffset;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Type;
use types::IRType;
use types::Shape;

#[derive(Clone)]
pub struct Shaped(MlirType);

impl Shaped {
    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_shaped() {
            eprint!("Cannot coerce type to shaped type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Shaped(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_shaped() {
            eprint!("Cannot coerce type to shaped type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn dim_size(&self, i: isize) -> i64 {
        do_unsafe!(mlirShapedTypeGetDimSize(self.0, i))
    }

    pub fn dynamic_size() -> i64 {
        do_unsafe!(mlirShapedTypeGetDynamicSize())
    }

    pub fn dynamic_stride_or_offset() -> i64 {
        do_unsafe!(mlirShapedTypeGetDynamicStrideOrOffset())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_element_type(&self) -> Type {
        Type::from(do_unsafe!(mlirShapedTypeGetElementType(self.0)))
    }

    pub fn is_static(&self) -> bool {
        do_unsafe!(mlirShapedTypeHasStaticShape(self.0))
    }

    pub fn is_dynamic_dim(&self, i: isize) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicDim(self.0, i))
    }

    pub fn is_dynamic_size(i: i64) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicSize(i))
    }

    pub fn is_dynamic_stride_or_offset(i: i64) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicStrideOrOffset(i))
    }

    pub fn has_rank(&self) -> bool {
        do_unsafe!(mlirShapedTypeHasRank(self.0))
    }

    pub fn rank(&self) -> Option<i64> {
        if self.has_rank() {
            Some(do_unsafe!(mlirShapedTypeGetRank(self.0)))
        } else {
            None
        }
    }

    /// TODO: Not all shapes allowed by MLIR are valid in the MLP AST
    pub fn unpack_shape(shape: &dyn Shape) -> (isize, Vec<i64>) {
        let r = shape.rank();
        let s = shape.get().iter().map(|d| *d as i64).collect();
        (r as isize, s)
    }
}

impl IRType for Shaped {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }
}
