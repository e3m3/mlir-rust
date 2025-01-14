// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirIntegerSetAttrGetTypeID;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IAttribute;
use dialects::affine::Set as AffineSet;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::StringBacked;
use ir::StringRef;
use ir::TypeID;

#[derive(Clone)]
pub struct IntegerSet(MlirAttribute);

impl IntegerSet {
    pub fn new(set: AffineSet) -> Self {
        let context = set.get_context();
        let s = StringBacked::from(format!("affine_set<{}>", set));
        Self::new_string(&context, &s.as_string_ref())
    }

    pub fn new_string(context: &Context, s: &StringRef) -> Self {
        let mut attr = Attribute::from_parse(context, s);
        Self::from_checked(*attr.get_mut())
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_integer_set() {
            eprintln!("Cannot coerce attribute to integer set attribute: {}", attr);
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

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirIntegerSetAttrGetTypeID()))
    }
}

impl From<MlirAttribute> for IntegerSet {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for IntegerSet {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for IntegerSet {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for IntegerSet {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
