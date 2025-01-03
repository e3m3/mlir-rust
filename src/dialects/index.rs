// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirOperation;

use std::fmt;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::IAttributeNamed;
use attributes::specialized::NamedBool;
use attributes::specialized::NamedIndex;
use attributes::specialized::SpecializedAttribute;
use dialects::IOp;
use dialects::IOperation;
use effects::MEFF_NO_MEMORY_EFFECT;
use effects::MemoryEffectList;
use exit_code::ExitCode;
use exit_code::exit;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::OperationState;
use ir::StringBacked;
use ir::Value;
use traits::Trait;
use types::IType;
use types::index::Index;
use types::integer::Integer as IntegerType;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct BoolValue(MlirAttribute);

#[derive(Clone)]
pub struct IndexValue(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy)]
pub enum Op {
    Add,
    And,
    BoolConstant,
    CastS,
    CastU,
    CeilDivS,
    CeilDivU,
    Cmp,
    Constant,
    DivS,
    DivU,
    FloorDivS,
    MaxS,
    MaxU,
    MinS,
    MinU,
    Mul,
    Or,
    RemS,
    RemU,
    Shl,
    ShrS,
    ShrU,
    SizeOf,
    Sub,
    XOr,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Add(MlirOperation);

#[derive(Clone)]
pub struct BoolConstant(MlirOperation);

#[derive(Clone)]
pub struct CastS(MlirOperation);

#[derive(Clone)]
pub struct CastU(MlirOperation);

#[derive(Clone)]
pub struct Constant(MlirOperation);

#[derive(Clone)]
pub struct SizeOf(MlirOperation);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl BoolValue {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IndexValue {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

///////////////////////////////
//  Enum Implementation
///////////////////////////////

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::And => "and",
            Op::BoolConstant => "bool.constant",
            Op::CastS => "casts",
            Op::CastU => "castu",
            Op::CeilDivS => "ceildivs",
            Op::CeilDivU => "ceildivu",
            Op::Cmp => "cmp",
            Op::Constant => "constant",
            Op::DivS => "divs",
            Op::DivU => "divu",
            Op::FloorDivS => "floordivs",
            Op::MaxS => "maxs",
            Op::MaxU => "maxu",
            Op::MinS => "mins",
            Op::MinU => "minu",
            Op::Mul => "mul",
            Op::Or => "or",
            Op::RemS => "rems",
            Op::RemU => "remu",
            Op::Shl => "shl",
            Op::ShrS => "shrs",
            Op::ShrU => "shru",
            Op::SizeOf => "sizeof",
            Op::Sub => "sub",
            Op::XOr => "xor",
        }
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

/// TODO: Not working for the index dialect.
fn get_dialect(context: &Context) -> Dialect {
    match context.load_dialect("index") {
        Some(d) => d,
        None => {
            eprintln!("Expected index dialect to be registered in context");
            exit(ExitCode::DialectError);
        }
    }
}

impl Add {
    pub fn new(context: &Context, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if !lhs.get_type().is_index() {
            eprintln!("Expected index type for lhs operand of add operation");
            exit(ExitCode::DialectError);
        }
        if !rhs.get_type().is_index() {
            eprintln!("Expected index type for rhs operand of add operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::Add.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl BoolConstant {
    pub fn new(context: &Context, value: &BoolValue, loc: &Location) -> Self {
        let t = IntegerType::new(context, 1).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::BoolConstant.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[value.as_named_attribute()]);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl CastS {
    pub fn new_index(context: &Context, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_integer() {
            eprintln!("Expected integer type for operand of casts operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::CastS.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_integer(t: &IntegerType, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_index() {
            eprintln!("Expected index type for operand of casts operation");
            exit(ExitCode::DialectError);
        }
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::CastS.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl CastU {
    pub fn new_index(context: &Context, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_integer() {
            eprintln!("Expected integer type for operand of castu operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::CastU.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_integer(t: &IntegerType, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_index() {
            eprintln!("Expected index type for operand of castu operation");
            exit(ExitCode::DialectError);
        }
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::CastU.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Constant {
    pub fn new(context: &Context, value: &IndexValue, loc: &Location) -> Self {
        let t = Index::new(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::Constant.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[value.as_named_attribute()]);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl SizeOf {
    pub fn new(context: &Context, loc: &Location) -> Self {
        let t = Index::new(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::SizeOf.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl From<MlirOperation> for Add {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Add {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Add.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Add
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for BoolConstant {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for BoolConstant {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::BoolConstant.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::BoolConstant
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::ConstantLike]
    }
}

SpecializedAttribute!("value" = impl NamedBool for BoolValue {});

impl From<MlirOperation> for CastS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for CastS {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CastS.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::CastS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for CastU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for CastU {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CastU.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::CastU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for Constant {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Constant {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Constant.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Constant
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::ConstantLike]
    }
}

SpecializedAttribute!("value" = impl NamedIndex for IndexValue {});

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for SizeOf {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for SizeOf {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::SizeOf.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::SizeOf
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Add => "AddOp",
            Op::And => "AndOp",
            Op::BoolConstant => "BoolConstantOp",
            Op::CastS => "CastSOp",
            Op::CastU => "CastUOp",
            Op::CeilDivS => "CeilDivSOp",
            Op::CeilDivU => "CeilDivUOp",
            Op::Cmp => "CmpOp",
            Op::Constant => "ConstantOp",
            Op::DivS => "DivSOp",
            Op::DivU => "DivUOp",
            Op::FloorDivS => "FloorDivSOp",
            Op::MaxS => "MaxSOp",
            Op::MaxU => "MaxUOp",
            Op::MinS => "MinSOp",
            Op::MinU => "MinUOp",
            Op::Mul => "MulOp",
            Op::Or => "OrOp",
            Op::RemS => "RemSOp",
            Op::RemU => "RemUOp",
            Op::Shl => "ShlOp",
            Op::ShrS => "ShrSOp",
            Op::ShrU => "ShrUOp",
            Op::SizeOf => "SizeOfOp",
            Op::Sub => "SubOp",
            Op::XOr => "XOrOp",
        })
    }
}
