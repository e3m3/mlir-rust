// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;

use std::cmp;
use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::integer::Integer as IntegerAttr;
use attributes::specialized::NamedBool;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedMemoryLayout;
use attributes::specialized::NamedMemorySpace;
use attributes::specialized::NamedString;
use attributes::strided_layout::StridedLayout;
use dialects::affine::Map as AffineMap;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Context;
use ir::StringBacked;
use types::integer::Integer as IntegerType;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct DefaultMemorySpace(MlirAttribute);

#[derive(Clone)]
pub struct Dimension(MlirAttribute);

#[derive(Clone)]
pub struct IntegerMemorySpace(MlirAttribute);

#[derive(Clone)]
pub struct MemoryLayout(MlirAttribute);

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

#[derive(Clone, Copy, Default, PartialEq)]
pub enum SymbolVisibilityKind {
    #[default]
    None,
    Private,
}

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl DefaultMemorySpace {
    pub fn new() -> Self {
        <Self as NamedMemorySpace>::new_none()
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IntegerMemorySpace {
    pub fn new(context: &Context, n: i64) -> Self {
        const WIDTH: usize = 64;
        let t = IntegerType::new(context, WIDTH);
        let attr = IntegerAttr::new(&t, n);
        <Self as NamedMemorySpace>::new_integer(&attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Dimension {
    pub fn new(context: &Context, n: i64) -> Self {
        const WIDTH: usize = 64;
        <Self as NamedInteger>::new(context, n, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl MemoryLayout {
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
            SymbolVisibilityKind::None => None,
            SymbolVisibilityKind::Private => {
                let s = StringBacked::from(k.to_string());
                Some(<Self as NamedString>::new(context, &s.as_string_ref()))
            }
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

impl Default for DefaultMemorySpace {
    fn default() -> Self {
        Self::new()
    }
}

impl From<MlirAttribute> for DefaultMemorySpace {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for DefaultMemorySpace {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for DefaultMemorySpace {
    fn get_name() -> &'static str {
        "memorySpace"
    }
}

impl NamedMemorySpace for DefaultMemorySpace {
    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_none() {
            eprintln!("Expected default memory space to be none memory space type");
            exit(ExitCode::DialectError);
        }
        attr_
    }
}

impl cmp::PartialEq for DefaultMemorySpace {
    fn eq(&self, rhs: &Self) -> bool {
        <Self as NamedMemorySpace>::eq(self, rhs)
    }
}

impl From<MlirAttribute> for IntegerMemorySpace {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for IntegerMemorySpace {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for IntegerMemorySpace {
    fn get_name() -> &'static str {
        "memorySpace"
    }
}

impl NamedMemorySpace for IntegerMemorySpace {
    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_integer() {
            eprintln!("Expected integer memory space to be integer memory space type");
            exit(ExitCode::DialectError);
        }
        attr_
    }
}

impl cmp::PartialEq for IntegerMemorySpace {
    fn eq(&self, rhs: &Self) -> bool {
        <Self as NamedMemorySpace>::eq(self, rhs)
    }
}

impl From<MlirAttribute> for Dimension {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for Dimension {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for Dimension {
    fn get_name() -> &'static str {
        "dim"
    }
}

impl NamedInteger for Dimension {}

impl From<MlirAttribute> for MemoryLayout {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl From<AffineMap> for MemoryLayout {
    fn from(map: AffineMap) -> Self {
        Self::from(&map)
    }
}

impl From<&AffineMap> for MemoryLayout {
    fn from(map: &AffineMap) -> Self {
        Self::new_affine_map(map)
    }
}

impl From<StridedLayout> for MemoryLayout {
    fn from(layout: StridedLayout) -> Self {
        Self::from(&layout)
    }
}

impl From<&StridedLayout> for MemoryLayout {
    fn from(layout: &StridedLayout) -> Self {
        Self::new_strided_layout(layout)
    }
}

impl IAttribute for MemoryLayout {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for MemoryLayout {
    fn get_name() -> &'static str {
        "layout"
    }
}

impl NamedMemoryLayout for MemoryLayout {}

impl From<MlirAttribute> for NonTemporal {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for NonTemporal {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for NonTemporal {
    fn get_name() -> &'static str {
        "nontemporal"
    }
}

impl NamedBool for NonTemporal {}

impl From<MlirAttribute> for OperandSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for OperandSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for OperandSegmentSizes {
    fn get_name() -> &'static str {
        "operandSegmentSizes"
    }
}

impl NamedI32DenseArray for OperandSegmentSizes {}

impl From<MlirAttribute> for ResultSegmentSizes {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for ResultSegmentSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for ResultSegmentSizes {
    fn get_name() -> &'static str {
        "resultSegmentSizes"
    }
}

impl NamedI32DenseArray for ResultSegmentSizes {}

impl From<MlirAttribute> for StaticOffsets {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for StaticOffsets {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for StaticOffsets {
    fn get_name() -> &'static str {
        "static_offsets"
    }
}

impl NamedI64DenseArray for StaticOffsets {}

impl From<MlirAttribute> for StaticSizes {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for StaticSizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for StaticSizes {
    fn get_name() -> &'static str {
        "static_sizes"
    }
}

impl NamedI64DenseArray for StaticSizes {}

impl From<MlirAttribute> for StaticStrides {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for StaticStrides {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for StaticStrides {
    fn get_name() -> &'static str {
        "static_strides"
    }
}

impl NamedI64DenseArray for StaticStrides {}

impl From<MlirAttribute> for SymbolName {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for SymbolName {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IAttributeNamed for SymbolName {
    fn get_name() -> &'static str {
        "sym_name"
    }
}

impl NamedString for SymbolName {}

impl From<MlirAttribute> for SymbolVisibility {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl FromStr for SymbolVisibilityKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(SymbolVisibilityKind::None),
            "private" => Ok(SymbolVisibilityKind::Private),
            _ => Err(format!("Invalid symbol visibility kind: {}", s)),
        }
    }
}

impl IAttribute for SymbolVisibility {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IAttributeNamed for SymbolVisibility {
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
            SymbolVisibilityKind::None => "none",
            SymbolVisibilityKind::Private => "private",
        })
    }
}
