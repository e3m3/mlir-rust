// Copyright 2024, Giordano Salvador
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
use ir::Type;
use ir::TypeID;
use types::IType;
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

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_ranked_tensor() {
            eprint!("Cannot coerce type to ranked tensor type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        RankedTensor(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(self.0)
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_encoding(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirRankedTensorTypeGetEncoding(self.0)))
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
        s.rank().unwrap_or(0) == s_other.rank().unwrap_or(0)
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

impl IType for RankedTensor {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
