// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirSparseElementsAttribute;
use mlir::mlirSparseElementsAttrGetIndices;
use mlir::mlirSparseElementsAttrGetTypeID;
use mlir::mlirSparseElementsAttrGetValues;
use mlir::MlirAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IRAttribute;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::TypeID;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct SparseElements(MlirAttribute);

impl SparseElements {
    pub fn new(t: &Shaped, indices: &Attribute, values: &Attribute) -> Self {
        if !indices.is_dense_elements_int() || !values.is_dense_elements() {
            eprintln!("Wrong type(s) for sparse elements indices or values");
            exit(ExitCode::IRError);
        }
        Self::from(do_unsafe!(mlirSparseElementsAttribute(*t.get(), *indices.get(), *values.get())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let attr_ = Attribute::from(attr);
        if !attr_.is_sparse_elements() {
            eprint!("Cannot coerce attribute to sparse elements attribute: ");
            attr_.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        SparseElements(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_indices(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirSparseElementsAttrGetIndices(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirSparseElementsAttrGetTypeID()))
    }

    pub fn get_values(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirSparseElementsAttrGetValues(self.0)))
    }
}

impl IRAttribute for SparseElements {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
