// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirFlatSymbolRefAttrGet;
use mlir::mlirFlatSymbolRefAttrGetValue;
use mlir::mlirSymbolRefAttrGet;
use mlir::mlirSymbolRefAttrGetLeafReference;
use mlir::mlirSymbolRefAttrGetNestedReference;
use mlir::mlirSymbolRefAttrGetNumNestedReferences;
use mlir::mlirSymbolRefAttrGetRootReference;
use mlir::mlirSymbolRefAttrGetTypeID;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
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
        Self::from(do_unsafe!(mlirFlatSymbolRefAttrGet(*context.get(), *sym.get())))
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

    pub fn get_nested_reference(&self, i: isize) -> Attribute {
        if i >= self.num_nested_references() || i < 0 {
            eprintln!("Index '{}' out of bounds for nested reference of symbol ref", i);
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
            Some(StringRef::from(do_unsafe!(mlirFlatSymbolRefAttrGetValue(self.0))))
        } else {
            None
        }
    }

    pub fn num_nested_references(&self) -> isize {
        do_unsafe!(mlirSymbolRefAttrGetNumNestedReferences(self.0))
    }
}

impl IRAttribute for SymbolRef {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }
}
