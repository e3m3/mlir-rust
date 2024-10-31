// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirUnmanagedDenseBoolResourceElementsAttrGet;
use mlir::mlirDenseBoolResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseDoubleResourceElementsAttrGet;
use mlir::mlirDenseDoubleResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseFloatResourceElementsAttrGet;
use mlir::mlirDenseFloatResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseInt8ResourceElementsAttrGet;
use mlir::mlirDenseInt8ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseInt16ResourceElementsAttrGet;
use mlir::mlirDenseInt16ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseInt32ResourceElementsAttrGet;
use mlir::mlirDenseInt32ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseInt64ResourceElementsAttrGet;
use mlir::mlirDenseInt64ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseUInt8ResourceElementsAttrGet;
use mlir::mlirDenseUInt8ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseUInt16ResourceElementsAttrGet;
use mlir::mlirDenseUInt16ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseUInt32ResourceElementsAttrGet;
use mlir::mlirDenseUInt32ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseUInt64ResourceElementsAttrGet;
use mlir::mlirDenseUInt64ResourceElementsAttrGetValue;
use mlir::mlirUnmanagedDenseResourceElementsAttrGet;
use mlir::MlirAttribute;

use std::ffi::c_int;
use std::ffi::c_void;
use std::fmt;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::StringRef;
use types::shaped::Shaped;

pub type DeleterFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, usize);

#[derive(Clone)]
pub struct DenseResourceElements(MlirAttribute, Layout);

#[derive(Clone,Copy,PartialEq)]
pub enum Layout {
    Bool,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    Inferred,
    U8,
    U16,
    U32,
    U64,
}

impl DenseResourceElements {
    pub fn new(
        t: &Shaped,
        name: &StringRef,
        data: &mut [c_void],
        user_data: &mut [c_void],
        alignment: usize,
        is_mut: bool,
        deleter: Option<DeleterFn>,
    ) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            data.as_mut_ptr(),
            data.len(),
            alignment,
            is_mut,
            deleter,
            user_data.as_mut_ptr(),
        )), Layout::Inferred)
    }

    pub fn new_bool(t: &Shaped, name: &StringRef, elements: &[bool]) -> Self {
        let e: Vec<c_int> = elements.iter().map(|e| *e as c_int).collect();
        Self::from(do_unsafe!(mlirUnmanagedDenseBoolResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            e.len() as isize,
            e.as_ptr(),
        )), Layout::Bool)
    }

    pub fn new_f32(t: &Shaped, name: &StringRef, elements: &[f32]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseFloatResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::F32)
    }

    pub fn new_f64(t: &Shaped, name: &StringRef, elements: &[f64]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseDoubleResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::F64)
    }

    pub fn new_i8(t: &Shaped, name: &StringRef, elements: &[i8]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseInt8ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I8)
    }

    pub fn new_i16(t: &Shaped, name: &StringRef, elements: &[i16]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseInt16ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I16)
    }

    pub fn new_i32(t: &Shaped, name: &StringRef, elements: &[i32]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseInt32ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I32)
    }

    pub fn new_i64(t: &Shaped, name: &StringRef, elements: &[i64]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseInt64ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I64)
    }

    pub fn new_u8(t: &Shaped, name: &StringRef, elements: &[u8]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U8)
    }

    pub fn new_u16(t: &Shaped, name: &StringRef, elements: &[u16]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U16)
    }

    pub fn new_u32(t: &Shaped, name: &StringRef, elements: &[u32]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U32)
    }

    pub fn new_u64(t: &Shaped, name: &StringRef, elements: &[u64]) -> Self {
        Self::from(do_unsafe!(mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
            *t.get(),
            *name.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U64)
    }

    pub fn from(attr: MlirAttribute, layout: Layout) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_dense_elements_resource() {
            eprint!("Cannot coerce attribute to dense resource elements attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        DenseResourceElements(attr, layout)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_bool(&self, i: isize) -> bool {
        do_unsafe!(mlirDenseBoolResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_f32(&self, i: isize) -> f32 {
        do_unsafe!(mlirDenseFloatResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_f64(&self, i: isize) -> f64 {
        do_unsafe!(mlirDenseDoubleResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_i8(&self, i: isize) -> i8 {
        do_unsafe!(mlirDenseInt8ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_i16(&self, i: isize) -> i16 {
        do_unsafe!(mlirDenseInt16ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_i32(&self, i: isize) -> i32 {
        do_unsafe!(mlirDenseInt32ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_i64(&self, i: isize) -> i64 {
        do_unsafe!(mlirDenseInt64ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_layout(&self) -> Layout {
        self.1
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_u8(&self, i: isize) -> u8 {
        do_unsafe!(mlirDenseUInt8ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_u16(&self, i: isize) -> u16 {
        do_unsafe!(mlirDenseUInt16ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_u32(&self, i: isize) -> u32 {
        do_unsafe!(mlirDenseUInt32ResourceElementsAttrGetValue(self.0, i))
    }

    pub fn get_u64(&self, i: isize) -> u64 {
        do_unsafe!(mlirDenseUInt64ResourceElementsAttrGetValue(self.0, i))
    }
}

impl IRAttribute for DenseResourceElements {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Layout::Bool        => "bool",
            Layout::F32         => "f32",
            Layout::F64         => "f64",
            Layout::I8          => "i8",
            Layout::I16         => "i16",
            Layout::I32         => "i32",
            Layout::I64         => "i64",
            Layout::Inferred    => "inferred",
            Layout::U8          => "u8",
            Layout::U16         => "u16",
            Layout::U32         => "u32",
            Layout::U64         => "u64",
        })
    }
}

