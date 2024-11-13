// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use crate::attributes;

use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::NamedI64DenseArray;

#[derive(Clone)]
pub struct OperandSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct ResultSegmentSizes(MlirAttribute);

impl OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl From<MlirAttribute> for OperandSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        OperandSegmentSizes(attr)
    }
}

impl IRAttribute for OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for OperandSegmentSizes {
    fn get_name() -> &'static str {
        "operand_segment_sizes"
    }
}

impl NamedI64DenseArray for OperandSegmentSizes {}

impl ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl From<MlirAttribute> for ResultSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        ResultSegmentSizes(attr)
    }
}

impl IRAttribute for ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for ResultSegmentSizes {
    fn get_name() -> &'static str {
        "result_segment_sizes"
    }
}

impl NamedI64DenseArray for ResultSegmentSizes {}
