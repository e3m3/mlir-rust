// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirMemRefTypeContiguousGet;
use mlir::mlirMemRefTypeContiguousGetChecked;
use mlir::mlirMemRefTypeGet;
use mlir::mlirMemRefTypeGetAffineMap;
use mlir::mlirMemRefTypeGetChecked;
use mlir::mlirMemRefTypeGetLayout;
use mlir::mlirMemRefTypeGetMemorySpace;
use mlir::mlirMemRefTypeGetStridesAndOffset;
use mlir::mlirMemRefTypeGetTypeID;
use mlir::MlirType;

use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use dialects::affine;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Location;
use ir::Shape;
use ir::Type;
use ir::TypeID;
use types::IRType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct MemRef(MlirType);

pub struct StridesAndOffset {
    strides: Box<[i64]>,
    offset: Box<[i64]>,
}

impl MemRef {
    pub fn new(shape: &dyn Shape, t: &Type, layout: &Attribute, memory_space: &Attribute) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirMemRefTypeGet(
            *t.get(),
            r,
            s.as_ptr(),
            *layout.get(),
            *memory_space.get(),
        )))
    }

    pub fn new_checked(
        shape: &dyn Shape,
        t: &Type,
        layout: &Attribute,
        memory_space: &Attribute,
        loc: &Location
    ) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirMemRefTypeGetChecked(
            *loc.get(),
            *t.get(),
            r,
            s.as_ptr(),
            *layout.get(),
            *memory_space.get(),
        )))
    }

    pub fn new_contiguous(shape: &dyn Shape, t: &Type, memory_space: &Attribute) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirMemRefTypeContiguousGet(*t.get(), r, s.as_ptr(), *memory_space.get())))
    }

    pub fn new_contiguous_checked(
        shape: &dyn Shape,
        t: &Type,
        memory_space: &Attribute,
        loc: &Location
    ) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirMemRefTypeContiguousGetChecked(
            *loc.get(),
            *t.get(),
            r,
            s.as_ptr(),
            *memory_space.get(),
        )))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_mem_ref() {
            eprint!("Cannot coerce type to mem ref type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        MemRef(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_mem_ref() {
            eprint!("Cannot coerce type to mem ref type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Self::from(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(self.0)
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_affine_map(&self) -> affine::Map {
        affine::Map::from(do_unsafe!(mlirMemRefTypeGetAffineMap(self.0)))
    }

    pub fn get_layout(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirMemRefTypeGetLayout(self.0)))
    }

    pub fn get_memory_space(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirMemRefTypeGetMemorySpace(self.0)))
    }

    pub fn get_strides_and_offset(&self) -> StridesAndOffset {
        let rank = self.as_shaped().rank().unwrap() as usize;
        let mut strides = vec![0; rank];
        let mut offset = vec![0; rank];
        do_unsafe!(mlirMemRefTypeGetStridesAndOffset(self.0, strides.as_mut_ptr(), offset.as_mut_ptr()));
        StridesAndOffset::new(strides.into_boxed_slice(), offset.into_boxed_slice())
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirMemRefTypeGetTypeID()))
    }
}

impl IRType for MemRef {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }
}

impl StridesAndOffset {
    pub fn new(strides: Box<[i64]>, offset: Box<[i64]>) -> Self {
        if strides.len() != offset.len() {
            eprintln!("Mismatched lengths for strides and offset");
            exit(ExitCode::IRError);
        }
        StridesAndOffset{strides, offset}
    }

    pub fn get_offset(&self) -> &[i64] {
        &self.offset
    }

    pub fn get_strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.strides.len()
    }
}
