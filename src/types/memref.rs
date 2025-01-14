// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirType;
use mlir_sys::mlirMemRefTypeContiguousGet;
use mlir_sys::mlirMemRefTypeContiguousGetChecked;
use mlir_sys::mlirMemRefTypeGet;
use mlir_sys::mlirMemRefTypeGetAffineMap;
use mlir_sys::mlirMemRefTypeGetChecked;
use mlir_sys::mlirMemRefTypeGetLayout;
use mlir_sys::mlirMemRefTypeGetMemorySpace;
use mlir_sys::mlirMemRefTypeGetStridesAndOffset;
use mlir_sys::mlirMemRefTypeGetTypeID;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::specialized::NamedMemoryLayout;
use attributes::specialized::NamedMemorySpace;
use attributes::strided_layout::StridedLayout;
use dialects::affine::Map as AffineMap;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Location;
use ir::LogicalResult;
use ir::Shape;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::shaped::NewElementType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct MemRef(MlirType);

impl MemRef {
    pub fn new(
        shape: &dyn Shape,
        t: &Type,
        layout: &impl NamedMemoryLayout,
        memory_space: &impl NamedMemorySpace,
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeGet(
            *t.get(),
            r,
            s.as_ptr(),
            *layout.as_attribute().get(),
            *memory_space.get(),
        )))
    }

    pub fn new_checked(
        shape: &dyn Shape,
        t: &Type,
        layout: &impl NamedMemoryLayout,
        memory_space: &impl NamedMemorySpace,
        loc: &Location,
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeGetChecked(
            *loc.get(),
            *t.get(),
            r,
            s.as_ptr(),
            *layout.as_attribute().get(),
            *memory_space.get(),
        )))
    }

    pub fn new_contiguous(
        shape: &dyn Shape,
        t: &Type,
        memory_space: &impl NamedMemorySpace,
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirMemRefTypeContiguousGet(
            *t.get(),
            r,
            s.as_ptr(),
            *memory_space.get()
        )))
    }

    pub fn new_contiguous_checked(
        shape: &dyn Shape,
        t: &Type,
        memory_space: &impl NamedMemorySpace,
        loc: &Location,
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

    pub fn from_type(t: &Type) -> Self {
        if !t.is_memref() {
            eprintln!("Cannot coerce type to mem ref type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(*self.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_affine_map(&self) -> AffineMap {
        AffineMap::from(do_unsafe!(mlirMemRefTypeGetAffineMap(*self.get())))
    }

    pub fn get_layout<T: From<MlirAttribute>>(&self) -> T {
        T::from(self.get_layout_attribute())
    }

    pub fn get_layout_attribute(&self) -> MlirAttribute {
        do_unsafe!(mlirMemRefTypeGetLayout(*self.get()))
    }

    pub fn get_matching_suffix<L: NamedMemoryLayout, S: NamedMemorySpace>(
        &self,
        other: &Self,
    ) -> Option<Self> {
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        s.get_matching_suffix(&s_other).map(|s_suffix| {
            let t = s.get_element_type();
            let l = self.get_layout::<L>();
            let m = self.get_memory_space::<S>();
            Self::new(&s_suffix, &t, &l, &m)
        })
    }

    pub fn get_memory_space<T: NamedMemorySpace>(&self) -> T {
        T::from_checked(self.get_memory_space_attribute())
    }

    pub fn get_memory_space_attribute(&self) -> MlirAttribute {
        do_unsafe!(mlirMemRefTypeGetMemorySpace(*self.get()))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_strided_layout(&self) -> Result<StridedLayout, String> {
        let rank = self
            .as_shaped()
            .rank()
            .unwrap_or(Err("Expected ranked memory reference".to_string())?)
            as usize;
        let mut strides = vec![0; rank];
        let mut offset = vec![0; rank];
        let result = LogicalResult::from(do_unsafe!(mlirMemRefTypeGetStridesAndOffset(
            *self.get(),
            strides.as_mut_ptr(),
            offset.as_mut_ptr(),
        )));
        if result.get_bool() {
            let context = self.get_context();
            let offset_ = offset.first().cloned().unwrap_or(Shaped::dynamic_size());
            Ok(StridedLayout::new(&context, offset_, &strides))
        } else {
            Err("Failed to get strides and offsets for memory reference".to_string())
        }
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirMemRefTypeGetTypeID()))
    }
}

impl From<MlirType> for MemRef {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for MemRef {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for MemRef {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for MemRef {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl NewElementType for MemRef {
    fn new_element_type(other: &Self, t: &Type) -> Self {
        let layout = other.get_layout_attribute();
        let memory_space = other.get_memory_space_attribute();
        let (r, s) = other.as_shaped().unpack();
        Self::from(do_unsafe!(mlirMemRefTypeGet(
            *t.get(),
            r,
            s.as_ptr(),
            layout,
            memory_space,
        )))
    }
}
