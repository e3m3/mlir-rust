// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::mlirSparseElementsAttrGetIndices;
use mlir_sys::mlirSparseElementsAttrGetTypeID;
use mlir_sys::mlirSparseElementsAttrGetValues;
use mlir_sys::mlirSparseElementsAttribute;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use exit_code::ExitCode;
use exit_code::exit;
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
        Self::from(do_unsafe!(mlirSparseElementsAttribute(
            *t.get(),
            *indices.get(),
            *values.get()
        )))
    }

    pub fn from_checked(attr_: MlirAttribute) -> Self {
        let attr = Attribute::from(attr_);
        if !attr.is_sparse_elements() {
            eprintln!(
                "Cannot coerce attribute to sparse elements attribute: {}",
                attr
            );
            exit(ExitCode::IRError);
        }
        Self::from(attr_)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_indices(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirSparseElementsAttrGetIndices(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirSparseElementsAttrGetTypeID()))
    }

    pub fn get_values(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirSparseElementsAttrGetValues(*self.get())))
    }
}

impl From<MlirAttribute> for SparseElements {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for SparseElements {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for SparseElements {
    fn from(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }
}

impl IAttribute for SparseElements {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}
