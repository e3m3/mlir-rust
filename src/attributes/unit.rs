// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::mlirUnitAttrGet;
use mlir_sys::mlirUnitAttrGetTypeID;
use mlir_sys::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::TypeID;

#[derive(Clone)]
pub struct Unit(MlirAttribute);

impl Unit {
    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirUnitAttrGet(*context.get())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_unit() {
            eprint!("Cannot coerce attribute to unit attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Unit(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirUnitAttrGetTypeID()))
    }
}

impl IRAttribute for Unit {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
