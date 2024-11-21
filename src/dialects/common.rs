// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::ir;

use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::NamedBool;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedString;
use ir::Context;
use ir::StringBacked;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct NonTemporal(MlirAttribute);

#[derive(Clone)]
pub struct OperandSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct ResultSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct SymbolName(MlirAttribute);

#[derive(Clone)]
pub struct SymbolVisibility(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[derive(Clone,Copy,Default,PartialEq)]
pub enum SymbolVisibilityKind {
    #[default]
    None,
    Private,
}

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl NonTemporal {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl SymbolName {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl SymbolVisibility {
    pub fn new(context: &Context, k: SymbolVisibilityKind) -> Option<Self> {
        match k {
            SymbolVisibilityKind::None      => None,
            SymbolVisibilityKind::Private   => {
                let s = StringBacked::from_string(&k.to_string());
                Some(<Self as NamedString>::new(context, &s.as_string_ref()))
            },
        }
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl From<MlirAttribute> for NonTemporal {
    fn from(attr: MlirAttribute) -> Self {
        NonTemporal(attr)
    }
}

impl IRAttribute for NonTemporal {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for NonTemporal {
    fn get_name() -> &'static str {
        "nontemporal"
    }
}

impl NamedBool for NonTemporal {}

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

impl From<MlirAttribute> for SymbolName {
    fn from(attr: MlirAttribute) -> Self {
        SymbolName(attr)
    }
}

impl IRAttribute for SymbolName {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for SymbolName {
    fn get_name() -> &'static str {
        "sym_name"
    }
}

impl NamedString for SymbolName {}

impl From<MlirAttribute> for SymbolVisibility {
    fn from(attr: MlirAttribute) -> Self {
        SymbolVisibility(attr)
    }
}

impl FromStr for SymbolVisibilityKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            ""          => Ok(SymbolVisibilityKind::None),
            "private"   => Ok(SymbolVisibilityKind::Private),
            _           => Err(format!("Invalid symbol visibility kind: {}", s)),
        }
    }
}

impl IRAttribute for SymbolVisibility {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for SymbolVisibility {
    fn get_name() -> &'static str {
        "sym_visibility"
    }
}

impl NamedString for SymbolVisibility {}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for SymbolVisibilityKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            SymbolVisibilityKind::None      => "none",
            SymbolVisibilityKind::Private   => "private",
        })
    }
}
