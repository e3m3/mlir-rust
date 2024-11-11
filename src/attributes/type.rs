// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirTypeAttrGet;
use mlir::mlirTypeAttrGetTypeID;
use mlir::mlirTypeAttrGetValue;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::TypeID;

#[derive(Clone)]
pub struct Type(MlirAttribute);

impl Type {
    pub fn new(t: &ir::Type) -> Self {
        Self::from(do_unsafe!(mlirTypeAttrGet(*t.get())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_type() {
            eprint!("Cannot coerce attribute to type attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Type(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type(&self) -> ir::Type {
        ir::Type::from(do_unsafe!(mlirTypeAttrGetValue(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirTypeAttrGetTypeID()))
    }
}

impl IRAttribute for Type {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
