// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirStridedLayoutAttrGet;
use mlir::mlirStridedLayoutAttrGetOffset;
use mlir::mlirStridedLayoutAttrGetNumStrides;
use mlir::mlirStridedLayoutAttrGetStride;
use mlir::mlirStridedLayoutAttrGetTypeID;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::TypeID;

#[derive(Clone)]
pub struct StridedLayout(MlirAttribute);

impl StridedLayout {
    pub fn new(context: &Context, offset: i64, strides: &[i64]) -> Self {
        Self::from(do_unsafe!(mlirStridedLayoutAttrGet(
            *context.get(),
            offset,
            strides.len() as isize,
            strides.as_ptr(),
        )))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_strided_layout() {
            eprint!("Cannot coerce attribute to strided layout attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        StridedLayout(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_offset(&self) -> i64 {
        do_unsafe!(mlirStridedLayoutAttrGetOffset(self.0))
    }

    pub fn get_stride(&self, i: isize) -> i64 {
        do_unsafe!(mlirStridedLayoutAttrGetStride(self.0, i))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirStridedLayoutAttrGetTypeID()))
    }

    pub fn num_strides(&self) -> isize {
        do_unsafe!(mlirStridedLayoutAttrGetNumStrides(self.0))
    }
}

impl IRAttribute for StridedLayout {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
