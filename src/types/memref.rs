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

use std::cmp;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::specialized::NamedMemorySpace;
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
    pub fn new(
        shape: &dyn Shape,
        t: &Type,
        layout: &Attribute,
        memory_space: &impl NamedMemorySpace,
    ) -> Self {
        let (r, s) = shape.unpack();
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
        memory_space: &impl NamedMemorySpace,
        loc: &Location
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeGetChecked(
            *loc.get(),
            *t.get(),
            r,
            s.as_ptr(),
            *layout.get(),
            *memory_space.get(),
        )))
    }

    pub fn new_contiguous(shape: &dyn Shape, t: &Type, memory_space: &impl NamedMemorySpace) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeContiguousGet(*t.get(), r, s.as_ptr(), *memory_space.get())))
    }

    pub fn new_contiguous_checked(
        shape: &dyn Shape,
        t: &Type,
        memory_space: &impl NamedMemorySpace,
        loc: &Location
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeContiguousGetChecked(
            *loc.get(),
            *t.get(),
            r,
            s.as_ptr(),
            *memory_space.get(),
        )))
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_mem_ref() {
            eprint!("Cannot coerce type to mem ref type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        MemRef(*t.get())
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

    pub fn get_matching_suffix<T: NamedMemorySpace>(&self, other: &Self) -> Option<Self> {
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        s.get_matching_suffix(&s_other).map(|s_suffix| {
            let t = s.get_element_type();
            let l = self.get_layout();
            let m = self.get_memory_space::<T>();
            Self::new(&s_suffix, &t, &l, &m)
        })
    }

    pub fn get_memory_space<T: NamedMemorySpace>(&self) -> T {
        T::from_checked(do_unsafe!(mlirMemRefTypeGetMemorySpace(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_strides_and_offset(&self) -> StridesAndOffset {
        let rank = self.as_shaped().rank().unwrap_or(0) as usize;
        let mut strides = vec![0; rank];
        let mut offset = vec![0; rank];
        do_unsafe!(mlirMemRefTypeGetStridesAndOffset(self.0, strides.as_mut_ptr(), offset.as_mut_ptr()));
        StridesAndOffset::new(&strides, &offset)
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirMemRefTypeGetTypeID()))
    }
}

impl IRType for MemRef {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl StridesAndOffset {
    pub fn new(strides: &[i64], offset: &[i64]) -> Self {
        if strides.len() != offset.len() {
            eprintln!("Mismatched lengths for strides and offset");
            exit(ExitCode::IRError);
        }
        StridesAndOffset{
            strides: strides.to_vec().into_boxed_slice(),
            offset: offset.to_vec().into_boxed_slice(),
        }
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

impl cmp::PartialEq for StridesAndOffset {
    fn eq(&self, rhs: &Self) -> bool {
        self.get_offset() == rhs.get_offset() && self.get_strides() == rhs.get_strides()
    }
}
