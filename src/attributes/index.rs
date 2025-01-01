// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;

use crate::attributes;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
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
        Self::from_checked(*attr.get())
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_index() {
            eprintln!("Cannot coerce attribute to index attribute type: {}", attr);
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    /// NOTE:   No getters for index attribute are currently provided by the C API.
    ///         As a hack, grab the ID from a freshly parsed index attribute.
    pub fn get_type_id() -> TypeID {
        let context = Context::new();
        let attr = Self::new(&context, 0).as_attribute();
        attr.get_type_id()
    }
}

impl From<MlirAttribute> for Index {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Index {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Index {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for Index {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
