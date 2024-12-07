// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirIndexTypeGetTypeID;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::ir;
use crate::exit_code;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::StringBacked;
use ir::TypeID;

#[derive(Clone)]
pub struct Index(MlirAttribute);

impl Index {
    /// NOTE: No getters for index attribute are currently provided by the C API.
    /// As a hack, construct a string and parse it as an attribute.
    pub fn new(context: &Context, value: i64) -> Self {
        let data = StringBacked::from(format!("{} : index", value));
        let attr = Attribute::from_parse(context, &data.as_string_ref());
        Self::from(*attr.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_index() {
            eprint!("Cannot coerce attribute to index attribute type: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Index(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    /// NOTE: No getters for index attribute are currently provided by the C API.
    /// As a hack, use the type ID for for an index type.
    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIndexTypeGetTypeID()))
    }
}

impl IRAttribute for Index {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
