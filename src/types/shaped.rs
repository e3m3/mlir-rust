// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirShapedTypeGetDimSize;
use mlir_sys::mlirShapedTypeGetDynamicSize;
use mlir_sys::mlirShapedTypeGetDynamicStrideOrOffset;
use mlir_sys::mlirShapedTypeGetElementType;
use mlir_sys::mlirShapedTypeGetRank;
use mlir_sys::mlirShapedTypeHasRank;
use mlir_sys::mlirShapedTypeHasStaticShape;
use mlir_sys::mlirShapedTypeIsDynamicDim;
use mlir_sys::mlirShapedTypeIsDynamicSize;
use mlir_sys::mlirShapedTypeIsDynamicStrideOrOffset;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Shape;
use ir::ShapeImpl;
use ir::Type;
use types::GetWidth;
use types::IType;
use types::memref::MemRef;
use types::ranked_tensor::RankedTensor;
use types::unranked_memref::UnrankedMemRef;
use types::unranked_tensor::UnrankedTensor;
use types::vector::Vector;

/// Copies the shape of the given Shaped type, returning a new Shaped type with
/// the element type given.
/// The resulting multi-dimensional Shaped type will be of the same type as the Shaped
/// type given (e.g., fn(tensor<10xi32>, f32) -> tensor<10xf32>).
pub trait NewElementType {
    fn new_element_type(other: &Self, t: &Type) -> Self;
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum ShapedKind {
    RankedMemRef,
    RankedTensor,
    UnrankedMemRef,
    UnrankedTensor,
    Vector,
}

#[derive(Clone)]
pub struct Shaped(MlirType);

impl Shaped {
    pub fn from_type(t: &Type) -> Self {
        if !t.is_shaped() {
            eprintln!("Cannot coerce type to shaped type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn dim_size(&self, i: isize) -> i64 {
        do_unsafe!(mlirShapedTypeGetDimSize(*self.get(), i))
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
        Type::from(do_unsafe!(mlirShapedTypeGetElementType(*self.get())))
    }

    pub fn get_kind(&self) -> ShapedKind {
        let t = self.as_type();
        if t.is_vector() {
            ShapedKind::Vector
        } else if t.is_unranked_memref() {
            ShapedKind::UnrankedMemRef
        } else if t.is_unranked_tensor() {
            ShapedKind::UnrankedTensor
        } else if t.is_ranked_tensor() {
            ShapedKind::RankedTensor
        } else if t.is_memref() {
            ShapedKind::RankedMemRef
        } else {
            eprintln!("Unexpected shaped type: {}", t);
            exit(ExitCode::DialectError);
        }
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_matching_suffix(&self, other: &Self) -> Option<ShapeImpl<Vec<i64>>> {
        if !self.has_matching_suffix(other) {
            return None;
        }
        let mut v: Vec<i64> = Vec::new();
        let mut n = -1 + self.rank().unwrap_or(-1) as isize;
        let mut n_other = -1 + other.rank().unwrap_or(-1) as isize;
        loop {
            if n < 0 || n_other < 0 {
                break;
            }
            let size = self.dim_size(n);
            let size_other = other.dim_size(n_other);
            if size == size_other {
                v.push(size);
            }
            n -= 1;
            n_other -= 1;
        }
        Some(ShapeImpl::from(v.into_iter().rev().collect::<Vec<i64>>()))
    }

    pub fn get_width(&self) -> Option<usize> {
        self.get_element_type().get_width()
    }

    pub fn has_matching_element_type_width(&self, other: &Self) -> bool {
        self.get_width() == other.get_width()
    }

    pub fn has_matching_suffix(&self, other: &Self) -> bool {
        if self.get_element_type() != other.get_element_type() {
            return false;
        }
        if !self.has_rank() || !other.has_rank() {
            return false;
        }
        let n = -1 + self.rank().unwrap_or(-1) as isize;
        let n_other = -1 + other.rank().unwrap_or(-1) as isize;
        n >= 0 && n_other >= 0 && self.dim_size(n) == other.dim_size(n_other)
    }

    pub fn has_rank(&self) -> bool {
        do_unsafe!(mlirShapedTypeHasRank(*self.get()))
    }

    pub fn has_dynamic_dims(&self) -> bool {
        if let Some(n) = self.num_dynamic_dims() {
            n > 0
        } else {
            false
        }
    }

    pub fn is_static(&self) -> bool {
        do_unsafe!(mlirShapedTypeHasStaticShape(*self.get()))
    }

    pub fn is_dynamic_dim(&self, i: isize) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicDim(*self.get(), i))
    }

    pub fn is_dynamic_size(i: i64) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicSize(i))
    }

    pub fn is_dynamic_stride_or_offset(i: i64) -> bool {
        do_unsafe!(mlirShapedTypeIsDynamicStrideOrOffset(i))
    }

    pub fn rank(&self) -> Option<i64> {
        if self.has_rank() {
            Some(do_unsafe!(mlirShapedTypeGetRank(*self.get())))
        } else {
            None
        }
    }

    /// Can only be computed if the shaped is statically sized with no dynamically sized dimensions.
    pub fn num_elements(&self) -> Option<i64> {
        if self.is_static() && !self.has_dynamic_dims() {
            self.rank()
                .map(|rank| (0..rank).fold(1, |acc, i| acc * self.dim_size(i as isize)))
        } else {
            None
        }
    }

    /// Can only be computed if the shaped is ranked.
    pub fn num_dynamic_dims(&self) -> Option<i64> {
        self.rank().map(|rank| {
            (0..rank).fold(0, |acc, i| {
                acc + if self.is_dynamic_dim(i as isize) {
                    1
                } else {
                    0
                }
            })
        })
    }
}

impl From<MlirType> for Shaped {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Shaped {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Shaped {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Shaped {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl NewElementType for Shaped {
    fn new_element_type(other: &Self, t: &Type) -> Self {
        match other.get_kind() {
            ShapedKind::RankedMemRef => {
                let other_ = MemRef::from(*other.get());
                MemRef::new_element_type(&other_, t).as_shaped()
            }
            ShapedKind::RankedTensor => {
                let other_ = RankedTensor::from(*other.get());
                RankedTensor::new_element_type(&other_, t).as_shaped()
            }
            ShapedKind::UnrankedMemRef => {
                let other_ = UnrankedMemRef::from(*other.get());
                UnrankedMemRef::new_element_type(&other_, t).as_shaped()
            }
            ShapedKind::UnrankedTensor => {
                let other_ = UnrankedTensor::from(*other.get());
                UnrankedTensor::new_element_type(&other_, t).as_shaped()
            }
            ShapedKind::Vector => {
                let other_ = Vector::from(*other.get());
                Vector::new_element_type(&other_, t).as_shaped()
            }
        }
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
