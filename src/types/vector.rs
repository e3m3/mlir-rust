// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirVectorTypeGet;
use mlir::mlirVectorTypeGetChecked;
use mlir::mlirVectorTypeGetScalable;
use mlir::mlirVectorTypeGetScalableChecked;
use mlir::mlirVectorTypeGetTypeID;
use mlir::mlirVectorTypeIsDimScalable;
use mlir::mlirVectorTypeIsScalable;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Location;
use ir::Shape;
use ir::Type;
use ir::TypeID;
use types::IRType;
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
        Self::from(do_unsafe!(mlirVectorTypeGetChecked(*loc.get(), r, s.as_ptr(), *t.get())))
    }

    pub fn new_checked_scalable(
        shape: &dyn Shape,
        t: &Type,
        is_scalable: &[bool],
        loc: &Location
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
        let t_ = Type::from(t);
        if !t_.is_vector() {
            eprint!("Cannot coerce type to vector type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Vector(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_vector() {
            eprint!("Cannot coerce type to vector type: ");
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

impl IRType for Vector {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
