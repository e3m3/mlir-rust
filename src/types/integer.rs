// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirIntegerTypeGet;
use mlir_sys::mlirIntegerTypeGetTypeID;
use mlir_sys::mlirIntegerTypeGetWidth;
use mlir_sys::mlirIntegerTypeIsSigned;
use mlir_sys::mlirIntegerTypeIsSignless;
use mlir_sys::mlirIntegerTypeIsUnsigned;
use mlir_sys::mlirIntegerTypeSignedGet;
use mlir_sys::mlirIntegerTypeUnsignedGet;

use std::ffi::c_uint;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::ExitCode;
use exit_code::exit;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::IsPromotableTo;

#[derive(Clone)]
pub struct Integer(MlirType);

impl Integer {
    /// Same as `Self::new_signless(...)`.
    pub fn new(context: &Context, width: usize) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeGet(
            *context.get(),
            width as c_uint
        )))
    }

    pub fn new_bool(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeGet(*context.get(), 1)))
    }

    pub fn new_bool_unsigned(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeUnsignedGet(*context.get(), 1)))
    }

    pub fn new_signed(context: &Context, width: usize) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeSignedGet(
            *context.get(),
            width as c_uint
        )))
    }

    pub fn new_signless(context: &Context, width: usize) -> Self {
        Self::new(context, width)
    }

    pub fn new_unsigned(context: &Context, width: usize) -> Self {
        Self::from(do_unsafe!(mlirIntegerTypeUnsignedGet(
            *context.get(),
            width as c_uint
        )))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_integer() {
            eprintln!("Cannot coerce type to integer type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_width(&self) -> usize {
        do_unsafe!(mlirIntegerTypeGetWidth(*self.get())) as usize
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerTypeGetTypeID()))
    }

    pub fn is_bool(&self) -> bool {
        self.is_unsigned() && self.get_width() == 1
    }

    pub fn is_signed(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsSigned(*self.get()))
    }

    pub fn is_signless(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsSignless(*self.get()))
    }

    pub fn is_unsigned(&self) -> bool {
        do_unsafe!(mlirIntegerTypeIsUnsigned(*self.get()))
    }
}

impl From<MlirType> for Integer {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for Integer {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for Integer {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for Integer {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl IsPromotableTo<Integer> for Integer {
    fn is_promotable_to(&self, other: &Self) -> bool {
        self.get_width() <= other.get_width()
    }
}
