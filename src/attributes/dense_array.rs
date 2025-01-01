// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirDenseArrayAttrGetTypeID;
use mlir_sys::mlirDenseArrayGetNumElements;
use mlir_sys::mlirDenseBoolArrayGet;
use mlir_sys::mlirDenseBoolArrayGetElement;
use mlir_sys::mlirDenseF32ArrayGet;
use mlir_sys::mlirDenseF32ArrayGetElement;
use mlir_sys::mlirDenseF64ArrayGet;
use mlir_sys::mlirDenseF64ArrayGetElement;
use mlir_sys::mlirDenseI8ArrayGet;
use mlir_sys::mlirDenseI8ArrayGetElement;
use mlir_sys::mlirDenseI16ArrayGet;
use mlir_sys::mlirDenseI16ArrayGetElement;
use mlir_sys::mlirDenseI32ArrayGet;
use mlir_sys::mlirDenseI32ArrayGetElement;
use mlir_sys::mlirDenseI64ArrayGet;
use mlir_sys::mlirDenseI64ArrayGetElement;

use std::ffi::c_int;
use std::fmt;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::TypeID;

#[derive(Clone)]
pub struct DenseArray(MlirAttribute, Layout);

#[derive(Clone, Copy, PartialEq)]
pub enum Layout {
    Bool,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
}

impl DenseArray {
    pub fn new_bool(context: &Context, elements: &[bool]) -> Self {
        let e: Vec<c_int> = elements.iter().map(|a| *a as c_int).collect();
        Self::from((
            do_unsafe!(mlirDenseBoolArrayGet(
                *context.get(),
                e.len() as isize,
                e.as_ptr()
            )),
            Layout::Bool,
        ))
    }

    pub fn new_f32(context: &Context, elements: &[f32]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseF32ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::F32,
        ))
    }

    pub fn new_f64(context: &Context, elements: &[f64]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseF64ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::F64,
        ))
    }

    pub fn new_i8(context: &Context, elements: &[i8]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseI8ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::I8,
        ))
    }

    pub fn new_i16(context: &Context, elements: &[i16]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseI16ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::I16,
        ))
    }

    pub fn new_i32(context: &Context, elements: &[i32]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseI32ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::I32,
        ))
    }

    pub fn new_i64(context: &Context, elements: &[i64]) -> Self {
        Self::from((
            do_unsafe!(mlirDenseI64ArrayGet(
                *context.get(),
                elements.len() as isize,
                elements.as_ptr()
            )),
            Layout::I64,
        ))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let mut attr = Self::from((attr_, Layout::Bool)); // Unused layout
        attr.1 = if attr.is_bool() {
            Layout::Bool
        } else if attr.is_f32() {
            Layout::F32
        } else if attr.is_f64() {
            Layout::F64
        } else if attr.is_i8() {
            Layout::I8
        } else if attr.is_i16() {
            Layout::I16
        } else if attr.is_i32() {
            Layout::I32
        } else if attr.is_i64() {
            Layout::I64
        } else {
            eprintln!(
                "Unexpected dense array for attribute: {}",
                attr.as_attribute()
            );
            exit(ExitCode::IRError);
        };
        attr
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_element_bool(&self, i: isize) -> bool {
        if self.get_layout() != Layout::Bool {
            eprintln!("Dense array does not hold elements of type bool");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseBoolArrayGetElement(*self.get(), i))
    }

    pub fn get_element_f32(&self, i: isize) -> f32 {
        if self.get_layout() != Layout::F32 {
            eprintln!("Dense array does not hold elements of type f32");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseF32ArrayGetElement(*self.get(), i))
    }

    pub fn get_element_f64(&self, i: isize) -> f64 {
        if self.get_layout() != Layout::F64 {
            eprintln!("Dense array does not hold elements of type f64");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseF64ArrayGetElement(*self.get(), i))
    }

    pub fn get_element_i8(&self, i: isize) -> i8 {
        if self.get_layout() != Layout::I8 {
            eprintln!("Dense array does not hold elements of type i8");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseI8ArrayGetElement(*self.get(), i))
    }

    pub fn get_element_i16(&self, i: isize) -> i16 {
        if self.get_layout() != Layout::I16 {
            eprintln!("Dense array does not hold elements of type i16");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseI16ArrayGetElement(*self.get(), i))
    }

    pub fn get_element_i32(&self, i: isize) -> i32 {
        if self.get_layout() != Layout::I32 {
            eprintln!("Dense array does not hold elements of type i32");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseI32ArrayGetElement(*self.get(), i))
    }

    pub fn get_element_i64(&self, i: isize) -> i64 {
        if self.get_layout() != Layout::I64 {
            eprintln!("Dense array does not hold elements of type i64");
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirDenseI64ArrayGetElement(*self.get(), i))
    }

    pub fn get_layout(&self) -> Layout {
        self.1
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirDenseArrayAttrGetTypeID()))
    }

    pub fn is(&self, layout: Layout) -> bool {
        self.get_layout() == layout
            && match layout {
                Layout::Bool => self.as_attribute().is_dense_array_bool(),
                Layout::F32 => self.as_attribute().is_dense_array_f32(),
                Layout::F64 => self.as_attribute().is_dense_array_f64(),
                Layout::I8 => self.as_attribute().is_dense_array_i8(),
                Layout::I16 => self.as_attribute().is_dense_array_i16(),
                Layout::I32 => self.as_attribute().is_dense_array_i32(),
                Layout::I64 => self.as_attribute().is_dense_array_i64(),
            }
    }

    pub fn is_bool(&self) -> bool {
        self.as_attribute().is_dense_array_bool()
    }

    pub fn is_f32(&self) -> bool {
        self.as_attribute().is_dense_array_f32()
    }

    pub fn is_f64(&self) -> bool {
        self.as_attribute().is_dense_array_f64()
    }

    pub fn is_i8(&self) -> bool {
        self.as_attribute().is_dense_array_i8()
    }

    pub fn is_i16(&self) -> bool {
        self.as_attribute().is_dense_array_i16()
    }

    pub fn is_i32(&self) -> bool {
        self.as_attribute().is_dense_array_i32()
    }

    pub fn is_i64(&self) -> bool {
        self.as_attribute().is_dense_array_i64()
    }

    pub fn num_elements(&self) -> isize {
        do_unsafe!(mlirDenseArrayGetNumElements(*self.get()))
    }
}

impl From<MlirAttribute> for DenseArray {
    fn from(attr: MlirAttribute) -> Self {
        Self::from_checked(attr)
    }
}

impl From<(MlirAttribute, Layout)> for DenseArray {
    fn from((attr, layout): (MlirAttribute, Layout)) -> Self {
        Self(attr, layout)
    }
}

impl From<Attribute> for DenseArray {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for DenseArray {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for DenseArray {
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
            Layout::Bool => "bool",
            Layout::F32 => "f32",
            Layout::F64 => "f64",
            Layout::I8 => "i8",
            Layout::I16 => "i16",
            Layout::I32 => "i32",
            Layout::I64 => "i64",
        })
    }
}
