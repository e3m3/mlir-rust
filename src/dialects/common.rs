// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use std::ffi::c_uint;
use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::ir;

use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::NamedBool;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedString;
use ir::Context;
use ir::StringBacked;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct Dimension(MlirAttribute);

#[derive(Clone)]
pub struct NonTemporal(MlirAttribute);

#[derive(Clone)]
pub struct OperandSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct ResultSegmentSizes(MlirAttribute);

#[derive(Clone)]
pub struct StaticOffsets(MlirAttribute);

#[derive(Clone)]
pub struct StaticSizes(MlirAttribute);

#[derive(Clone)]
pub struct StaticStrides(MlirAttribute);

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

impl Dimension {
    pub fn new(context: &Context, n: i64) -> Self {
        const WIDTH: c_uint = 64;
        <Self as NamedInteger>::new(context, n, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

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

impl StaticOffsets {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticSizes {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticStrides {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
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

impl From<MlirAttribute> for Dimension {
    fn from(attr: MlirAttribute) -> Self {
        Dimension(attr)
    }
}

impl IRAttribute for Dimension {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Dimension {
    fn get_name() -> &'static str {
        "dim"
    }
}

impl NamedInteger for Dimension {}

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
        "operandSegmentSizes"
    }
}

impl NamedI32DenseArray for OperandSegmentSizes {}

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
        "resultSegmentSizes"
    }
}

impl NamedI32DenseArray for ResultSegmentSizes {}

impl From<MlirAttribute> for StaticOffsets {
    fn from(attr: MlirAttribute) -> Self {
        StaticOffsets(attr)
    }
}

impl IRAttribute for StaticOffsets {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticOffsets {
    fn get_name() -> &'static str {
        "static_offsets"
    }
}

impl NamedI64DenseArray for StaticOffsets {}

impl From<MlirAttribute> for StaticSizes {
    fn from(attr: MlirAttribute) -> Self {
        StaticSizes(attr)
    }
}

impl IRAttribute for StaticSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticSizes {
    fn get_name() -> &'static str {
        "static_sizes"
    }
}

impl NamedI64DenseArray for StaticSizes {}

impl From<MlirAttribute> for StaticStrides {
    fn from(attr: MlirAttribute) -> Self {
        StaticStrides(attr)
    }
}

impl IRAttribute for StaticStrides {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticStrides {
    fn get_name() -> &'static str {
        "static_strides"
    }
}

impl NamedI64DenseArray for StaticStrides {}

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
