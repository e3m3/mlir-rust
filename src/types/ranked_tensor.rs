// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirRankedTensorTypeGet;
use mlir::mlirRankedTensorTypeGetChecked;
use mlir::mlirRankedTensorTypeGetEncoding;
use mlir::mlirRankedTensorTypeGetTypeID;
use mlir::MlirType;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Location;
use ir::Type;
use ir::TypeID;
use types::IRType;
use types::Shape;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct RankedTensor(MlirType);

impl RankedTensor {
    pub fn new(shape: &dyn Shape, t: &Type, encoding: &Attribute) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirRankedTensorTypeGet(r, s.as_ptr(), *t.get(), *encoding.get())))
    }

    pub fn new_checked(shape: &dyn Shape, t: &Type, encoding: &Attribute, loc: &Location) -> Self {
        let (r, s) = Shaped::unpack_shape(shape);
        Self::from(do_unsafe!(mlirRankedTensorTypeGetChecked(
            *loc.get(),
            r,
            s.as_ptr(),
            *t.get(),
            *encoding.get(),
        )))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_ranked_tensor() {
            eprint!("Cannot coerce type to ranked tensor type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        RankedTensor(t)
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_ranked_tensor() {
            eprint!("Cannot coerce type to ranked tensor type: ");
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

    pub fn get_encoding(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirRankedTensorTypeGetEncoding(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirRankedTensorTypeGetTypeID()))
    }
}

impl IRType for RankedTensor {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }
}
