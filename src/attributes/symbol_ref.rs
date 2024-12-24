// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirFlatSymbolRefAttrGet;
use mlir_sys::mlirFlatSymbolRefAttrGetValue;
use mlir_sys::mlirSymbolRefAttrGet;
use mlir_sys::mlirSymbolRefAttrGetLeafReference;
use mlir_sys::mlirSymbolRefAttrGetNestedReference;
use mlir_sys::mlirSymbolRefAttrGetNumNestedReferences;
use mlir_sys::mlirSymbolRefAttrGetRootReference;
use mlir_sys::mlirSymbolRefAttrGetTypeID;

use std::cmp;
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
use ir::TypeID;

#[derive(Clone)]
pub struct SymbolRef(MlirAttribute);

impl SymbolRef {
    pub fn new(context: &Context, sym: &StringRef, refs: &[Attribute]) -> Self {
        let r: Vec<MlirAttribute> = refs.iter().map(|a| *a.get()).collect();
        Self::from(do_unsafe!(mlirSymbolRefAttrGet(
            *context.get(),
            *sym.get(),
            refs.len() as isize,
            r.as_ptr(),
        )))
    }

    pub fn new_flat(context: &Context, sym: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirFlatSymbolRefAttrGet(
            *context.get(),
            *sym.get()
        )))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_symbol_ref() {
            eprint!("Cannot coerce attribute to symbol reference attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        SymbolRef(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_leaf(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirSymbolRefAttrGetLeafReference(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_nested_reference(&self, i: isize) -> Attribute {
        if i >= self.num_nested_references() || i < 0 {
            eprintln!(
                "Index '{}' out of bounds for nested reference of symbol ref",
                i
            );
            exit(ExitCode::IRError);
        }
        Attribute::from(do_unsafe!(mlirSymbolRefAttrGetNestedReference(self.0, i)))
    }

    pub fn get_root(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirSymbolRefAttrGetRootReference(self.0)))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirSymbolRefAttrGetTypeID()))
    }

    /// If this is a flat symbol reference, return the referenced symbol.
    pub fn get_value(&self) -> Option<StringRef> {
        if self.as_attribute().is_flat_symbol_ref() {
            Some(StringRef::from(do_unsafe!(mlirFlatSymbolRefAttrGetValue(
                self.0
            ))))
        } else {
            None
        }
    }

    pub fn num_nested_references(&self) -> isize {
        do_unsafe!(mlirSymbolRefAttrGetNumNestedReferences(self.0))
    }
}

impl fmt::Display for SymbolRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self.get_value() {
            None => "".to_string(),
            Some(s) => s.to_string(),
        })
    }
}

impl IAttribute for SymbolRef {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl cmp::PartialEq for SymbolRef {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}
