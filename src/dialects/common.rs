// Copyright 2024-2025, Giordano Salvador
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

use attributes::integer::Integer as IntegerAttr;
use attributes::specialized::NamedBool;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedMemoryLayout;
use attributes::specialized::NamedMemorySpace;
use attributes::specialized::NamedString;
use attributes::specialized::SpecializedAttribute;
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

SpecializedAttribute!("memorySpace" = impl NamedMemorySpace for DefaultMemorySpace {
    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_none() {
            eprintln!("Expected default memory space to be none memory space type");
            exit(ExitCode::DialectError);
        }
        attr_
    }
});

impl cmp::PartialEq for DefaultMemorySpace {
    fn eq(&self, rhs: &Self) -> bool {
        <Self as NamedMemorySpace>::eq(self, rhs)
    }
}

SpecializedAttribute!("memorySpace" = impl NamedMemorySpace for IntegerMemorySpace {
    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_integer() {
            eprintln!("Expected integer memory space to be integer memory space type");
            exit(ExitCode::DialectError);
        }
        attr_
    }
});

impl cmp::PartialEq for IntegerMemorySpace {
    fn eq(&self, rhs: &Self) -> bool {
        <Self as NamedMemorySpace>::eq(self, rhs)
    }
}

SpecializedAttribute!("dim" = impl NamedInteger for Dimension {});

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

SpecializedAttribute!("layout" = impl NamedMemoryLayout for MemoryLayout {});

SpecializedAttribute!("nontemporal" = impl NamedBool for NonTemporal {});

SpecializedAttribute!("operandSegmentSizes" = impl NamedI32DenseArray for OperandSegmentSizes {});

SpecializedAttribute!("resultSegmentSizes" = impl NamedI32DenseArray for ResultSegmentSizes {});

SpecializedAttribute!("static_offsets" = impl NamedI64DenseArray for StaticOffsets {});

SpecializedAttribute!("static_sizes" = impl NamedI64DenseArray for StaticSizes {});

SpecializedAttribute!("static_strides" = impl NamedI64DenseArray for StaticStrides {});

SpecializedAttribute!("sym_name" = impl NamedString for SymbolName {});

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

SpecializedAttribute!("sym_visibility" = impl NamedString for SymbolVisibility {});

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
