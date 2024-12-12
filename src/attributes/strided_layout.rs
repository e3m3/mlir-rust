// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::mlirStridedLayoutAttrGet;
use mlir_sys::mlirStridedLayoutAttrGetOffset;
use mlir_sys::mlirStridedLayoutAttrGetNumStrides;
use mlir_sys::mlirStridedLayoutAttrGetStride;
use mlir_sys::mlirStridedLayoutAttrGetTypeID;
use mlir_sys::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::TypeID;
use types::shaped::Shaped;

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

    pub fn get_stride(&self, i: isize) -> Result<i64, String> {
        if i < 0 || i >= self.num_strides() {
            Err("Index is out of bounds for strides of memory reference".to_string())
        } else {
            Ok(do_unsafe!(mlirStridedLayoutAttrGetStride(self.0, i)))
        }
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirStridedLayoutAttrGetTypeID()))
    }

    pub fn is_dynamic_offset(&self) -> bool {
        self.get_offset() == Shaped::dynamic_size()
    }

    pub fn is_dynamic_stride(&self, i: isize) -> Result<bool, String> {
        if i < 0 || i >= self.num_strides() {
            Err("Index is out of bounds for strides of memory reference".to_string())
        } else {
            Ok(self.get_stride(i)? == Shaped::dynamic_size())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_strides() == 0
    }

    pub fn num_strides(&self) -> isize {
        do_unsafe!(mlirStridedLayoutAttrGetNumStrides(self.0))
    }

    pub fn rank(&self) -> isize {
        self.num_strides()
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
