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
use ir::Shape;
use ir::Type;
use types::IRType;

#[derive(Clone)]
pub struct Shaped(MlirType);

impl Shaped {
    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_shaped() {
            eprint!("Cannot coerce type to shaped type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Shaped(*t.get())
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

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
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
}

impl IRType for Shaped {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl Shape for Shaped {
    fn rank(&self) -> isize {
        self.rank().unwrap_or(-1) as isize
    }

    fn get(&self, i: isize) -> i64 {
        self.dim_size(i)
    }
}
