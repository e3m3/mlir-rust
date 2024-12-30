// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirStringAttrGet;
use mlir_sys::mlirStringAttrGetTypeID;
use mlir_sys::mlirStringAttrGetValue;
use mlir_sys::mlirStringAttrTypedGet;

use std::fmt;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::StringRef;
use ir::Type;
use ir::TypeID;

#[derive(Clone)]
pub struct String(MlirAttribute);

impl String {
    pub fn new(context: &Context, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirStringAttrGet(*context.get(), *s.get())))
    }

    pub fn new_typed(t: &Type, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirStringAttrTypedGet(*t.get(), *s.get())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_string() {
            eprint!("Cannot coerce attribute to string attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        String(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_string(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirStringAttrGetValue(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirStringAttrGetTypeID()))
    }
}

impl fmt::Display for String {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_string())
    }
}

impl IAttribute for String {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
