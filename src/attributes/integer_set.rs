// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirIntegerSetAttrGetTypeID;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::TypeID;

#[derive(Clone)]
pub struct IntegerSet(MlirAttribute);

impl IntegerSet {
    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_integer_set() {
            eprint!("Cannot coerce attribute to integer set attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        IntegerSet(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerSetAttrGetTypeID()))
    }
}

impl IRAttribute for IntegerSet {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
