// Copyright 2024-2025, Giordano Salvador
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
use attributes::specialized::NamedParsed;
use attributes::specialized::SpecializedAttribute;
use dialects::IOp;
use dialects::IOperation;
use dialects::OpRef;
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

#[derive(Clone)]
pub struct Predicate(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum CmpPredicate {
    EQ = 0,
    NE = 1,
    SLT = 2,
    SLE = 3,
    SGT = 4,
    SGE = 5,
    ULT = 6,
    ULE = 7,
    UGT = 8,
    UGE = 9,
}

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
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
    ShL,
    ShRS,
    ShRU,
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
pub struct And(MlirOperation);

#[derive(Clone)]
pub struct BoolConstant(MlirOperation);

#[derive(Clone)]
pub struct CastS(MlirOperation);

#[derive(Clone)]
pub struct CastU(MlirOperation);

#[derive(Clone)]
pub struct CeilDivS(MlirOperation);

#[derive(Clone)]
pub struct CeilDivU(MlirOperation);

#[derive(Clone)]
pub struct Cmp(MlirOperation);

#[derive(Clone)]
pub struct Constant(MlirOperation);

#[derive(Clone)]
pub struct DivS(MlirOperation);

#[derive(Clone)]
pub struct DivU(MlirOperation);

#[derive(Clone)]
pub struct FloorDivS(MlirOperation);

#[derive(Clone)]
pub struct MaxS(MlirOperation);

#[derive(Clone)]
pub struct MaxU(MlirOperation);

#[derive(Clone)]
pub struct MinS(MlirOperation);

#[derive(Clone)]
pub struct MinU(MlirOperation);

#[derive(Clone)]
pub struct Mul(MlirOperation);

#[derive(Clone)]
pub struct Or(MlirOperation);

#[derive(Clone)]
pub struct RemS(MlirOperation);

#[derive(Clone)]
pub struct RemU(MlirOperation);

#[derive(Clone)]
pub struct ShL(MlirOperation);

#[derive(Clone)]
pub struct ShRS(MlirOperation);

#[derive(Clone)]
pub struct ShRU(MlirOperation);

#[derive(Clone)]
pub struct SizeOf(MlirOperation);

#[derive(Clone)]
pub struct Sub(MlirOperation);

#[derive(Clone)]
pub struct XOr(MlirOperation);

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

impl Predicate {
    pub fn new(context: &Context, pred: CmpPredicate) -> Self {
        let string = StringBacked::from(format!("#index<cmp_predicate {}>", pred));
        <Self as NamedParsed>::new(context, &string.as_string_ref())
    }

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

impl CmpPredicate {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => CmpPredicate::EQ,
            1 => CmpPredicate::NE,
            2 => CmpPredicate::SLT,
            3 => CmpPredicate::SLE,
            4 => CmpPredicate::SGT,
            5 => CmpPredicate::SGE,
            6 => CmpPredicate::ULT,
            7 => CmpPredicate::ULE,
            8 => CmpPredicate::UGT,
            9 => CmpPredicate::UGE,
            _ => {
                eprintln!("Invalid value '{}' for CmpPredicate", n);
                exit(ExitCode::DialectError);
            }
        }
    }
}

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
            Op::ShL => "shl",
            Op::ShRS => "shrs",
            Op::ShRU => "shru",
            Op::SizeOf => "sizeof",
            Op::Sub => "sub",
            Op::XOr => "xor",
        }
    }
}

///////////////////////////////
//  Support
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

macro_rules! binary_operation {
    ($Op:ident) => {
        impl $Op {
            pub fn new(context: &Context, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
                let name = Op::$Op.get_name();
                if !lhs.get_type().is_index() {
                    eprintln!("Expected index type for lhs operand of {} operation", name);
                    exit(ExitCode::DialectError);
                }
                if !rhs.get_type().is_index() {
                    eprintln!("Expected index type for rhs operand of {} operation", name);
                    exit(ExitCode::DialectError);
                }
                let t = Index::new(context).as_type();
                let name = StringBacked::from(format!(
                    "{}.{}",
                    "index", // NOTE: See `get_dialect`.
                    name,
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
    };
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

binary_operation!(Add);

binary_operation!(And);

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

binary_operation!(CeilDivS);

binary_operation!(CeilDivU);

impl Cmp {
    pub fn new(
        context: &Context,
        pred: CmpPredicate,
        lhs: &Value,
        rhs: &Value,
        loc: &Location,
    ) -> Self {
        if !lhs.get_type().is_index() {
            eprintln!("Expected index type for lhs operand of cmp operation");
            exit(ExitCode::DialectError);
        }
        if !rhs.get_type().is_index() {
            eprintln!("Expected index type for rhs operand of cpm operation");
            exit(ExitCode::DialectError);
        }
        let t = IntegerType::new_bool(context).as_type();
        let name = StringBacked::from(format!(
            "{}.{}",
            "index", // NOTE: See `get_dialect`.
            Op::Cmp.get_name(),
        ));
        let attr = Predicate::new(context, pred).as_named_attribute();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr]);
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

binary_operation!(DivS);

binary_operation!(DivU);

binary_operation!(FloorDivS);

binary_operation!(MaxS);

binary_operation!(MaxU);

binary_operation!(MinS);

binary_operation!(MinU);

binary_operation!(Mul);

binary_operation!(Or);

binary_operation!(RemS);

binary_operation!(RemU);

binary_operation!(ShL);

binary_operation!(ShRS);

binary_operation!(ShRU);

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

binary_operation!(Sub);

binary_operation!(XOr);

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

    fn get_op(&self) -> OpRef {
        &Op::Add
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for And {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for And {
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
        Op::And.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::And
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

    fn get_op(&self) -> OpRef {
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

    fn get_op(&self) -> OpRef {
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

    fn get_op(&self) -> OpRef {
        &Op::CastU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for CeilDivS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for CeilDivS {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CeilDivS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::CeilDivS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for CeilDivU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for CeilDivU {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CeilDivU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::CeilDivU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
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

    fn get_op(&self) -> OpRef {
        &Op::Constant
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::ConstantLike]
    }
}

impl From<MlirOperation> for Cmp {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Cmp {
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
        Op::Cmp.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Cmp
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for DivS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivS {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::DivS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for DivU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivU {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::DivU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for FloorDivS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for FloorDivS {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::FloorDivS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::FloorDivS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

SpecializedAttribute!("value" = impl NamedIndex for IndexValue {});

impl From<MlirOperation> for MaxS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MaxS {
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
        Op::MaxS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::MaxS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for MaxU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MaxU {
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
        Op::MaxU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::MaxU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for MinS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MinS {
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
        Op::MinS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::MinS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for MinU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MinU {
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
        Op::MinU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::MinU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<MlirOperation> for Mul {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Mul {
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
        Op::Mul.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Mul
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

impl From<i32> for CmpPredicate {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for CmpPredicate {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for Or {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Or {
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
        Op::Or.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Or
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

SpecializedAttribute!("pred" = impl NamedParsed for Predicate {});

impl From<MlirOperation> for RemS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for RemS {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::RemS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::RemS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for RemU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for RemU {
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
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::RemU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::RemU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for ShL {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ShL {
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
        Op::ShL.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::ShL
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for ShRS {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ShRS {
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
        Op::ShRS.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::ShRS
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for ShRU {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ShRU {
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
        Op::ShRU.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::ShRU
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
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

    fn get_op(&self) -> OpRef {
        &Op::SizeOf
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for Sub {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Sub {
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
        Op::Sub.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Sub
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirOperation> for XOr {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for XOr {
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
        Op::XOr.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::XOr
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Commutative]
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for CmpPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CmpPredicate::EQ => "eq",
            CmpPredicate::NE => "ne",
            CmpPredicate::SLT => "slt",
            CmpPredicate::SLE => "sle",
            CmpPredicate::SGT => "sgt",
            CmpPredicate::SGE => "sge",
            CmpPredicate::ULT => "ult",
            CmpPredicate::ULE => "ule",
            CmpPredicate::UGT => "ugt",
            CmpPredicate::UGE => "uge",
        })
    }
}

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
            Op::ShL => "ShlOp",
            Op::ShRS => "ShrSOp",
            Op::ShRU => "ShrUOp",
            Op::SizeOf => "SizeOfOp",
            Op::Sub => "SubOp",
            Op::XOr => "XOrOp",
        })
    }
}
