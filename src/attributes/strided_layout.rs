// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirStridedLayoutAttrGet;
use mlir_sys::mlirStridedLayoutAttrGetNumStrides;
use mlir_sys::mlirStridedLayoutAttrGetOffset;
use mlir_sys::mlirStridedLayoutAttrGetStride;
use mlir_sys::mlirStridedLayoutAttrGetTypeID;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
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

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_strided_layout() {
            eprintln!(
                "Cannot coerce attribute to strided layout attribute: {}",
                attr
            );
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_offset(&self) -> i64 {
        do_unsafe!(mlirStridedLayoutAttrGetOffset(*self.get()))
    }

    pub fn get_stride(&self, i: isize) -> Result<i64, String> {
        if i < 0 || i >= self.num_strides() {
            Err("Index is out of bounds for strides of memory reference".to_string())
        } else {
            Ok(do_unsafe!(mlirStridedLayoutAttrGetStride(*self.get(), i)))
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
        do_unsafe!(mlirStridedLayoutAttrGetNumStrides(*self.get()))
    }

    pub fn rank(&self) -> isize {
        self.num_strides()
    }
}

impl From<MlirAttribute> for StridedLayout {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for StridedLayout {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for StridedLayout {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for StridedLayout {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
