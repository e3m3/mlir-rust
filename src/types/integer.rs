// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirIntegerTypeGet;
use mlir::mlirIntegerTypeGetWidth;
use mlir::mlirIntegerTypeGetTypeID;
use mlir::mlirIntegerTypeIsSigned;
use mlir::mlirIntegerTypeIsSignless;
use mlir::mlirIntegerTypeIsUnsigned;
use mlir::mlirIntegerTypeSignedGet;
use mlir::mlirIntegerTypeUnsignedGet;
use mlir::MlirType;

use std::ffi::c_uint;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Integer(MlirType);

impl Integer {
    pub fn new(context: &Context, width: c_uint) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeGet(*context.get(), width)))
    }

    pub fn new_bool(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeUnsignedGet(*context.get(), 1)))
    }

    pub fn new_signed(context: &Context, width: c_uint) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeSignedGet(*context.get(), width)))
    }

    pub fn new_signless(context: &Context, width: c_uint) -> Self {
        Self::new(context, width)
    }

    pub fn new_unsigned(context: &Context, width: c_uint) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeUnsignedGet(*context.get(), width)))
    }

    pub fn from(t: MlirType) -> Self {
        let t_ = Type::from(t);
        if !t_.is_integer() {
            eprint!("Cannot coerce type to integer type: ");
            t_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Integer(t)
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_width(&self) -> c_uint {
        do_unsafe!(mlirIntegerTypeGetWidth(self.0))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerTypeGetTypeID()))
    }

    pub fn is_bool(&self) -> bool {
        self.is_unsigned() && self.get_width() == 1 as c_uint
    }

    pub fn is_signed(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsSigned(self.0))
    }

    pub fn is_signless(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsSignless(self.0))
    }

    pub fn is_unsigned(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsUnsigned(self.0))
    }
}

impl IRType for Integer {
    fn as_type(&self) -> Type {
        Type::from(self.0)
    }

    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
