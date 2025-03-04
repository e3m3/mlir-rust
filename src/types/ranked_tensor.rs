// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirRankedTensorTypeGet;
use mlir_sys::mlirRankedTensorTypeGetChecked;
use mlir_sys::mlirRankedTensorTypeGetEncoding;
use mlir_sys::mlirRankedTensorTypeGetTypeID;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Location;
use ir::Shape;
use ir::ShapeImpl;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::shaped::NewElementType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct RankedTensor(MlirType);

impl RankedTensor {
    pub fn new(shape: &dyn Shape, t: &Type) -> Self {
        let (r, s) = shape.unpack();
        let encoding = Attribute::new();
        Self::from(do_unsafe!(mlirRankedTensorTypeGet(
            r,
            s.as_ptr(),
            *t.get(),
            *encoding.get()
        )))
    }

    pub fn new_encoded(shape: &dyn Shape, t: &Type, encoding: &Attribute) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirRankedTensorTypeGet(
            r,
            s.as_ptr(),
            *t.get(),
            *encoding.get()
        )))
    }

    pub fn new_checked(shape: &dyn Shape, t: &Type, encoding: &Attribute, loc: &Location) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirRankedTensorTypeGetChecked(
            *loc.get(),
            r,
            s.as_ptr(),
            *t.get(),
            *encoding.get(),
        )))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_ranked_tensor() {
            eprintln!("Cannot coerce type to ranked tensor type: {}", t);
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

    pub fn get_encoding(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirRankedTensorTypeGetEncoding(*self.get())))
    }

    pub fn get_matching_suffix(&self, other: &Self) -> Option<Self> {
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        s.get_matching_suffix(&s_other).map(|s_suffix| {
            let t = s.get_element_type();
            let e = self.get_encoding();
            Self::new_encoded(&s_suffix, &t, &e)
        })
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirRankedTensorTypeGetTypeID()))
    }

    pub fn has_matching_ranks(&self, other: &Self) -> bool {
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        s.rank().unwrap_or(-1) == s_other.rank().unwrap_or(-1)
    }

    pub fn has_matching_static_dimensions(&self, other: &Self) -> bool {
        if !self.has_matching_ranks(other) {
            return false;
        }
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        let rank = s.rank().unwrap_or(0);
        for i in 0..rank {
            let i_ = i as isize;
            if !s.is_dynamic_dim(i_)
                && !s_other.is_dynamic_dim(i_)
                && s.dim_size(i_) != s_other.dim_size(i_)
            {
                return false;
            }
        }
        true
    }
}

impl From<MlirType> for RankedTensor {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for RankedTensor {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for RankedTensor {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for RankedTensor {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl NewElementType for RankedTensor {
    fn new_element_type(other: &Self, t: &Type) -> Self {
        let s = ShapeImpl::from(other.as_shaped().to_vec());
        Self::new(&s, t)
    }
}
