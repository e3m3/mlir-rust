// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::mlirDenseElementsAttrBoolGet;
use mlir_sys::mlirDenseElementsAttrGetBoolValue;
use mlir_sys::mlirDenseElementsAttrBoolSplatGet;
use mlir_sys::mlirDenseElementsAttrGetBoolSplatValue;
use mlir_sys::mlirDenseElementsAttrBFloat16Get;
use mlir_sys::mlirDenseElementsAttrDoubleGet;
use mlir_sys::mlirDenseElementsAttrGetDoubleValue;
use mlir_sys::mlirDenseElementsAttrDoubleSplatGet;
use mlir_sys::mlirDenseElementsAttrGetDoubleSplatValue;
use mlir_sys::mlirDenseElementsAttrFloat16Get;
use mlir_sys::mlirDenseElementsAttrFloatGet;
use mlir_sys::mlirDenseElementsAttrGetFloatValue;
use mlir_sys::mlirDenseElementsAttrFloatSplatGet;
use mlir_sys::mlirDenseElementsAttrGetFloatSplatValue;
use mlir_sys::mlirDenseElementsAttrInt8Get;
use mlir_sys::mlirDenseElementsAttrGetInt8Value;
use mlir_sys::mlirDenseElementsAttrInt8SplatGet;
use mlir_sys::mlirDenseElementsAttrGetInt8SplatValue;
use mlir_sys::mlirDenseElementsAttrInt16Get;
use mlir_sys::mlirDenseElementsAttrGetInt16Value;
use mlir_sys::mlirDenseElementsAttrInt32Get;
use mlir_sys::mlirDenseElementsAttrGetInt32Value;
use mlir_sys::mlirDenseElementsAttrInt32SplatGet;
use mlir_sys::mlirDenseElementsAttrGetInt32SplatValue;
use mlir_sys::mlirDenseElementsAttrInt64Get;
use mlir_sys::mlirDenseElementsAttrGetInt64Value;
use mlir_sys::mlirDenseElementsAttrInt64SplatGet;
use mlir_sys::mlirDenseElementsAttrGetInt64SplatValue;
use mlir_sys::mlirDenseElementsAttrStringGet;
use mlir_sys::mlirDenseElementsAttrGetStringValue;
use mlir_sys::mlirDenseElementsAttrGetStringSplatValue;
use mlir_sys::mlirDenseElementsAttrUInt8Get;
use mlir_sys::mlirDenseElementsAttrGetUInt8Value;
use mlir_sys::mlirDenseElementsAttrUInt8SplatGet;
use mlir_sys::mlirDenseElementsAttrGetUInt8SplatValue;
use mlir_sys::mlirDenseElementsAttrUInt16Get;
use mlir_sys::mlirDenseElementsAttrGetUInt16Value;
use mlir_sys::mlirDenseElementsAttrUInt32Get;
use mlir_sys::mlirDenseElementsAttrGetUInt32Value;
use mlir_sys::mlirDenseElementsAttrUInt32SplatGet;
use mlir_sys::mlirDenseElementsAttrGetUInt32SplatValue;
use mlir_sys::mlirDenseElementsAttrUInt64Get;
use mlir_sys::mlirDenseElementsAttrGetUInt64Value;
use mlir_sys::mlirDenseElementsAttrUInt64SplatGet;
use mlir_sys::mlirDenseElementsAttrGetUInt64SplatValue;
use mlir_sys::mlirDenseElementsAttrGet;
use mlir_sys::mlirDenseElementsAttrGetRawData;
use mlir_sys::mlirDenseElementsAttrIsSplat;
use mlir_sys::mlirDenseElementsAttrRawBufferGet;
use mlir_sys::mlirDenseElementsAttrReshapeGet;
use mlir_sys::mlirDenseElementsAttrSplatGet;
use mlir_sys::mlirDenseElementsAttrGetSplatValue;
use mlir_sys::mlirDenseIntOrFPElementsAttrGetTypeID;
use mlir_sys::MlirAttribute;
use mlir_sys::MlirStringRef;

use std::ffi::c_int;
use std::ffi::c_void;
use std::fmt;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::elements::Elements;
use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::StringRef;
use ir::TypeID;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct DenseElements(MlirAttribute, Layout);

#[derive(Clone,Copy,PartialEq)]
pub enum Layout {
    Bool,
    BF16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    Inferred,
    String,
    U8,
    U16,
    U32,
    U64,
}

impl DenseElements {
    pub fn new(t: &Shaped, elements: &[Attribute]) -> Self {
        let e: Vec<MlirAttribute> = elements.iter().map(|a| *a.get()).collect();
        Self::from(do_unsafe!(mlirDenseElementsAttrGet(
            *t.get(),
            e.len() as isize,
            e.as_ptr(),
        )), Layout::Inferred)
    }

    /// # Safety
    /// May dereference raw pointer 'buffer'.
    pub unsafe fn new_raw(t: &Shaped, size: usize, buffer: *const c_void) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrRawBufferGet(*t.get(), size, buffer)), Layout::Inferred)
    }

    pub fn new_reshape(t: &Shaped, attr: &Self) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrReshapeGet(*attr.get(), *t.get())), Layout::Inferred)
    }

    pub fn new_splat(t: &Shaped, element: &Attribute) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrSplatGet(*t.get(), *element.get())), Layout::Inferred)
    }

    pub fn new_bool(t: &Shaped, elements: &[bool]) -> Self {
        let e: Vec<c_int> = elements.iter().map(|e| *e as c_int).collect();
        Self::from(do_unsafe!(mlirDenseElementsAttrBoolGet(
            *t.get(),
            e.len() as isize,
            e.as_ptr(),
        )), Layout::Bool)
    }

    pub fn new_bool_splat(t: &Shaped, element: bool) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrBoolSplatGet(*t.get(), element)), Layout::Bool)
    }

    pub fn new_bf16(t: &Shaped, elements: &[u16]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrBFloat16Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::BF16)
    }

    pub fn new_f16(t: &Shaped, elements: &[u16]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrFloat16Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::F16)
    }

    pub fn new_f32(t: &Shaped, elements: &[f32]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrFloatGet(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::F32)
    }

    pub fn new_f32_splat(t: &Shaped, element: f32) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrFloatSplatGet(*t.get(), element)), Layout::F32)
    }

    pub fn new_f64(t: &Shaped, elements: &[f64]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrDoubleGet(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::F64)
    }

    pub fn new_f64_splat(t: &Shaped, element: f64) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrDoubleSplatGet(*t.get(), element)), Layout::F64)
    }

    pub fn new_i8(t: &Shaped, elements: &[i8]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt8Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I8)
    }

    pub fn new_i8_splat(t: &Shaped, element: i8) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt8SplatGet(*t.get(), element)), Layout::I8)
    }

    pub fn new_i16(t: &Shaped, elements: &[i16]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt16Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I16)
    }

    pub fn new_i32(t: &Shaped, elements: &[i32]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt32Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I32)
    }

    pub fn new_i32_splat(t: &Shaped, element: i32) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt32SplatGet(*t.get(), element)), Layout::I32)
    }

    pub fn new_i64(t: &Shaped, elements: &[i64]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt64Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::I64)
    }

    pub fn new_i64_splat(t: &Shaped, element: i64) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrInt64SplatGet(*t.get(), element)), Layout::I64)
    }

    pub fn new_string(t: &Shaped, elements: &[StringRef]) -> Self {
        let mut e: Vec<MlirStringRef> = elements.iter().map(|e| *e.get()).collect();
        Self::from(do_unsafe!(mlirDenseElementsAttrStringGet(
            *t.get(),
            e.len() as isize,
            e.as_mut_ptr(),
        )), Layout::String)
    }

    pub fn new_u8(t: &Shaped, elements: &[u8]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt8Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U8)
    }

    pub fn new_u8_splat(t: &Shaped, element: u8) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt8SplatGet(*t.get(), element)), Layout::U8)
    }

    pub fn new_u16(t: &Shaped, elements: &[u16]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt16Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U16)
    }

    pub fn new_u32(t: &Shaped, elements: &[u32]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt32Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U32)
    }

    pub fn new_u32_splat(t: &Shaped, element: u32) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt32SplatGet(*t.get(), element)), Layout::U32)
    }

    pub fn new_u64(t: &Shaped, elements: &[u64]) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt64Get(
            *t.get(),
            elements.len() as isize,
            elements.as_ptr(),
        )), Layout::U64)
    }

    pub fn new_u64_splat(t: &Shaped, element: u64) -> Self {
        Self::from(do_unsafe!(mlirDenseElementsAttrUInt64SplatGet(*t.get(), element)), Layout::U64)
    }

    pub fn as_elements(&self) -> Elements {
        Elements::from(*self.get())
    }

    pub fn from(attr: MlirAttribute, layout: Layout) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_dense_elements() {
            eprint!("Cannot coerce attribute to dense elements attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        // TODO: Type-specific dense elements checks not working.
        //if match layout {
        //    Layout::Bool        => false,
        //    Layout::BF16        => attr_.is_dense_elements_float(),
        //    Layout::F16         => attr_.is_dense_elements_float(),
        //    Layout::F32         => attr_.is_dense_elements_float(),
        //    Layout::F64         => attr_.is_dense_elements_float(),
        //    Layout::I8          => attr_.is_dense_elements_int(),
        //    Layout::I16         => attr_.is_dense_elements_int(),
        //    Layout::I32         => attr_.is_dense_elements_int(),
        //    Layout::I64         => attr_.is_dense_elements_int(),
        //    Layout::Inferred    => false,
        //    Layout::String      => false,
        //    Layout::U8          => attr_.is_dense_elements_int(),
        //    Layout::U16         => attr_.is_dense_elements_int(),
        //    Layout::U32         => attr_.is_dense_elements_int(),
        //    Layout::U64         => attr_.is_dense_elements_int(),
        //} {
        //    eprint!("Cannot coerce attribute to dense elements layout '{}': ", layout);
        //    attr_.dump();
        //    eprintln!();
        //    exit(ExitCode::IRError);
        //}
        DenseElements(attr, layout)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_bool(&self, i: isize) -> bool {
        do_unsafe!(mlirDenseElementsAttrGetBoolValue(self.0, i))
    }

    pub fn get_bool_splat(&self) -> bool {
        do_unsafe!(mlirDenseElementsAttrGetBoolSplatValue(self.0)) != 0
    }

    pub fn get_f32(&self, i: isize) -> f32 {
        do_unsafe!(mlirDenseElementsAttrGetFloatValue(self.0, i))
    }

    pub fn get_f32_splat(&self) -> f32 {
        do_unsafe!(mlirDenseElementsAttrGetFloatSplatValue(self.0))
    }

    pub fn get_f64(&self, i: isize) -> f64 {
        do_unsafe!(mlirDenseElementsAttrGetDoubleValue(self.0, i))
    }

    pub fn get_f64_splat(&self) -> f64 {
        do_unsafe!(mlirDenseElementsAttrGetDoubleSplatValue(self.0))
    }

    pub fn get_i8(&self, i: isize) -> i8 {
        do_unsafe!(mlirDenseElementsAttrGetInt8Value(self.0, i))
    }

    pub fn get_i8_splat(&self) -> i8 {
        do_unsafe!(mlirDenseElementsAttrGetInt8SplatValue(self.0))
    }

    pub fn get_i16(&self, i: isize) -> i16 {
        do_unsafe!(mlirDenseElementsAttrGetInt16Value(self.0, i))
    }

    pub fn get_i32(&self, i: isize) -> i32 {
        do_unsafe!(mlirDenseElementsAttrGetInt32Value(self.0, i))
    }

    pub fn get_i32_splat(&self) -> i32 {
        do_unsafe!(mlirDenseElementsAttrGetInt32SplatValue(self.0))
    }

    pub fn get_i64(&self, i: isize) -> i64 {
        do_unsafe!(mlirDenseElementsAttrGetInt64Value(self.0, i))
    }

    pub fn get_i64_splat(&self) -> i64 {
        do_unsafe!(mlirDenseElementsAttrGetInt64SplatValue(self.0))
    }

    pub fn get_layout(&self) -> Layout {
        self.1
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_raw_data(&self) -> *const c_void {
        do_unsafe!(mlirDenseElementsAttrGetRawData(self.0))
    }

    pub fn get_splat(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirDenseElementsAttrGetSplatValue(self.0)))
    }

    pub fn get_string(&self, i: isize) -> StringRef {
        StringRef::from(do_unsafe!(mlirDenseElementsAttrGetStringValue(self.0, i)))
    }

    pub fn get_string_splat(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirDenseElementsAttrGetStringSplatValue(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirDenseIntOrFPElementsAttrGetTypeID()))
    }

    pub fn get_u8(&self, i: isize) -> u8 {
        do_unsafe!(mlirDenseElementsAttrGetUInt8Value(self.0, i))
    }

    pub fn get_u8_splat(&self) -> u8 {
        do_unsafe!(mlirDenseElementsAttrGetUInt8SplatValue(self.0))
    }

    pub fn get_u16(&self, i: isize) -> u16 {
        do_unsafe!(mlirDenseElementsAttrGetUInt16Value(self.0, i))
    }

    pub fn get_u32(&self, i: isize) -> u32 {
        do_unsafe!(mlirDenseElementsAttrGetUInt32Value(self.0, i))
    }

    pub fn get_u32_splat(&self) -> u32 {
        do_unsafe!(mlirDenseElementsAttrGetUInt32SplatValue(self.0))
    }

    pub fn get_u64(&self, i: isize) -> u64 {
        do_unsafe!(mlirDenseElementsAttrGetUInt64Value(self.0, i))
    }

    pub fn get_u64_splat(&self) -> u64 {
        do_unsafe!(mlirDenseElementsAttrGetUInt64SplatValue(self.0))
    }

    pub fn is_splat(&self) -> bool {
        do_unsafe!(mlirDenseElementsAttrIsSplat(self.0))
    }
}

impl IRAttribute for DenseElements {
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
            Layout::BF16        => "bf16",
            Layout::F16         => "f16",
            Layout::F32         => "f32",
            Layout::F64         => "f64",
            Layout::I8          => "i8",
            Layout::I16         => "i16",
            Layout::I32         => "i32",
            Layout::I64         => "i64",
            Layout::Inferred    => "inferred",
            Layout::String      => "string",
            Layout::U8          => "u8",
            Layout::U16         => "u16",
            Layout::U32         => "u32",
            Layout::U64         => "u64",
        })
    }
}
