// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirVectorTypeGet;
use mlir_sys::mlirVectorTypeGetChecked;
use mlir_sys::mlirVectorTypeGetScalable;
use mlir_sys::mlirVectorTypeGetScalableChecked;
use mlir_sys::mlirVectorTypeGetTypeID;
use mlir_sys::mlirVectorTypeIsDimScalable;
use mlir_sys::mlirVectorTypeIsScalable;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Location;
use ir::Shape;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct Vector(MlirType);

impl Vector {
    pub fn new(shape: &dyn Shape, t: &Type) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirVectorTypeGet(r, s.as_ptr(), *t.get())))
    }

    pub fn new_checked(shape: &dyn Shape, t: &Type, loc: &Location) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirVectorTypeGetChecked(
            *loc.get(),
            r,
            s.as_ptr(),
            *t.get()
        )))
    }

    pub fn new_checked_scalable(
        shape: &dyn Shape,
        t: &Type,
        is_scalable: &[bool],
        loc: &Location,
    ) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirVectorTypeGetScalableChecked(
            *loc.get(),
            r,
            s.as_ptr(),
            is_scalable.as_ptr(),
            *t.get(),
        )))
    }

    pub fn new_scalable(shape: &dyn Shape, t: &Type, is_scalable: &[bool]) -> Self {
        let (r, s) = shape.unpack();
        Self::from(do_unsafe!(mlirVectorTypeGetScalable(
            r,
            s.as_ptr(),
            is_scalable.as_ptr(),
            *t.get(),
        )))
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_vector() {
            eprint!("Cannot coerce type to vector type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Vector(*t.get())
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(self.0)
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_matching_suffix(&self, other: &Self) -> Option<Self> {
        let s = self.as_shaped();
        let s_other = other.as_shaped();
        s.get_matching_suffix(&s_other).map(|s_suffix| {
            let t = s.get_element_type();
            Self::new(&s_suffix, &t)
        })
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirVectorTypeGetTypeID()))
    }

    pub fn is_scalable(&self) -> bool {
        do_unsafe!(mlirVectorTypeIsScalable(self.0))
    }

    pub fn is_scalable_dim(&self, i: isize) -> bool {
        do_unsafe!(mlirVectorTypeIsDimScalable(self.0, i))
    }
}

impl IType for Vector {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
