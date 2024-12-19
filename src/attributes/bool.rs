// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirBoolAttrGet;
use mlir_sys::mlirBoolAttrGetValue;

use std::ffi::c_int;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;

#[derive(Clone)]
pub struct Bool(MlirAttribute);

impl Bool {
    pub fn new(context: &Context, value: bool) -> Self {
        Self::from(do_unsafe!(mlirBoolAttrGet(*context.get(), value as c_int)))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_bool() {
            eprint!("Cannot coerce attribute to bool attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Bool(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_value(&self) -> bool {
        do_unsafe!(mlirBoolAttrGetValue(self.0))
    }
}

impl IRAttribute for Bool {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
