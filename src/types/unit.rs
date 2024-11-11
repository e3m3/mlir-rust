// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirType;

use crate::attributes;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IRAttribute;
use attributes::unit::Unit as UnitAttr;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Unit(MlirType);

/// Unit type is exposed by the C API.
/// However, the unit attribute can return its type, so construct it through the attribute.
impl Unit {
    pub fn new(context: &Context) -> Self {
        Self::from(*UnitAttr::new(context).as_attribute().get_type().get())
    }

    pub fn from(t: MlirType) -> Self {
        Self::from_type(&Type::from(t))
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unit() {
            eprint!("Cannot coerce type to unit type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Unit(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        UnitAttr::get_type_id()
    }
}

impl IRType for Unit {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
