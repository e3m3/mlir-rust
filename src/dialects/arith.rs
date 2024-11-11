// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;
use mlir::MlirOperation;

use std::cmp;
use std::ffi::c_uint;
use std::fmt;

use crate::attributes;
use crate::dialects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::float::Float as FloatAttr;
use attributes::integer::Integer as IntegerAttr;
use attributes::IRAttribute;
use attributes::named::Named;
use dialects::DialectOp;
use dialects::DialectOperation;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Attribute;
use ir::Context;
use ir::Dialect;
use ir::Destroy;
use ir::Identifier;
use ir::Location;
use ir::Operation;
use ir::OperationState;
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::integer::Integer as IntegerType;
use types::IRType;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct FastMath(MlirAttribute);

#[derive(Clone)]
pub struct IntegerOverflow(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum AtomicRMWKind {
    AddF        = 0,
    AddI        = 1,
    Assign      = 2,
    MaximumF    = 3,
    MaxS        = 4,
    MaxU        = 5,
    MinimumF    = 6,
    MinS        = 7,
    MinU        = 8,
    MulF        = 9,
    MulI        = 10,
    OrI         = 11,
    AndI        = 12,
    MaxNumF     = 13,
    MinNumF     = 14,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum CmpFPPredicate {
    AlwaysFalse = 0,
    OEQ         = 1,
    OGT         = 2,
    OGE         = 3,
    OLT         = 4,
    OLE         = 5,
    ONE         = 6,
    ORD         = 7,
    UEQ         = 8,
    UGT         = 9,
    UGE         = 10,
    ULT         = 11,
    ULE         = 12,
    UNE         = 13,
    UNO         = 14,
    AlwaysTrue  = 15,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum CmpIPredicate {
    Eq  = 0,
    Ne  = 1,
    Slt = 2,
    Sle = 3,
    Sgt = 4,
    Sge = 5,
    Ult = 6,
    Ule = 7,
    Ugt = 8,
    Uge = 9,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum FastMathFlags {
    None        = 0,
    ReAssoc     = 1,
    NNaN        = 2,
    NInf        = 4,
    NSz         = 8,
    ARcp        = 16,
    Contract    = 32,
    AFn         = 64,
    Fast        = 127,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum IntegerOverflowFlags {
    None    = 0,
    NSW     = 1,
    NUW     = 2,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum RoundingMode {
    ToNearestEven   = 0,
    Downward        = 1,
    Upward          = 2,
    TowardZero      = 3,
    ToNearestAway   = 4,
}

#[derive(Clone,Copy,PartialEq)]
pub enum Op {
    AddF,
    AddI,
    AddUIExtended,
    AndI,
    Bitcast,
    CeilDivSI,
    CeilDivUI,
    CmpF,
    CmpI,
    Constant,
    DivF,
    DivSI,
    DivUI,
    ExtF,
    ExtSI,
    ExtUI,
    FloorDivSI,
    FPToSI,
    FPToUI,
    IndexCast,
    IndexCastUI,
    MaximumF,
    MaxNumF,
    MaxSI,
    MaxUI,
    MinimumF,
    MinNumF,
    MinSI,
    MinUI,
    MulF,
    MulI,
    MulSIExtended,
    MulUIExtended,
    NegF,
    OrI,
    RemF,
    RemSI,
    RemUI,
    Select,
    ShlI,
    ShrSI,
    ShrUI,
    SIToFP,
    SubF,
    SubI,
    TruncF,
    TruncI,
    UIToFP,
    XorI,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct AddF(MlirOperation);

#[derive(Clone)]
pub struct AddI(MlirOperation);

#[derive(Clone)]
pub struct AddUIExtended(MlirOperation);

#[derive(Clone)]
pub struct Constant(MlirOperation);

#[derive(Clone)]
pub struct DivF(MlirOperation);

#[derive(Clone)]
pub struct DivSI(MlirOperation);

#[derive(Clone)]
pub struct DivUI(MlirOperation);

#[derive(Clone)]
pub struct MulF(MlirOperation);

#[derive(Clone)]
pub struct MulI(MlirOperation);

#[derive(Clone)]
pub struct MulSIExtended(MlirOperation);

#[derive(Clone)]
pub struct MulUIExtended(MlirOperation);

#[derive(Clone)]
pub struct SubF(MlirOperation);

#[derive(Clone)]
pub struct SubI(MlirOperation);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl FastMath {
    pub fn new(context: &Context, flags: FastMathFlags) -> Self {
        const WIDTH: c_uint = 8; // Hardcode width to 8 bits; Attribute only accepts i64 though.
        let t = IntegerType::new_signless(context, WIDTH);
        let attr = IntegerAttr::new(&t.as_type(), flags as i64);
        Self::from(*attr.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        FastMath(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_integer_attribute(&self) -> IntegerAttr {
        IntegerAttr::from(self.0)
    }

    pub fn get_flags(&self) -> FastMathFlags {
        FastMathFlags::from_i64(self.get_integer_attribute().get_int())
    }
}

impl IntegerOverflow {
    pub fn new(context: &Context, flags: IntegerOverflowFlags) -> Self {
        const WIDTH: c_uint = 8; // Hardcode width to 8 bits; Attribute only accepts i64 though.
        let t = IntegerType::new_signless(context, WIDTH);
        let attr = IntegerAttr::new(&t.as_type(), flags as i64);
        Self::from(*attr.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        IntegerOverflow(attr)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_integer_attribute(&self) -> IntegerAttr {
        IntegerAttr::from(self.0)
    }

    pub fn get_flags(&self) -> IntegerOverflowFlags {
        IntegerOverflowFlags::from_i64(self.get_integer_attribute().get_int())
    }
}

///////////////////////////////
//  Enum Implementation
///////////////////////////////

impl AtomicRMWKind {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => AtomicRMWKind::AddF,
            1   => AtomicRMWKind::AddI,
            2   => AtomicRMWKind::Assign,
            3   => AtomicRMWKind::MaximumF,
            4   => AtomicRMWKind::MaxS,
            5   => AtomicRMWKind::MaxU,
            6   => AtomicRMWKind::MinimumF,
            7   => AtomicRMWKind::MinS,
            8   => AtomicRMWKind::MinU,
            9   => AtomicRMWKind::MulF,
            10  => AtomicRMWKind::MulI,
            11  => AtomicRMWKind::OrI,
            12  => AtomicRMWKind::AndI,
            13  => AtomicRMWKind::MaxNumF,
            14  => AtomicRMWKind::MinNumF,
            _   => {
                eprintln!("Invalid value for AtomicRMWKind: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl CmpFPPredicate {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => CmpFPPredicate::AlwaysFalse,
            1   => CmpFPPredicate::OEQ,
            2   => CmpFPPredicate::OGT,
            3   => CmpFPPredicate::OGE,
            4   => CmpFPPredicate::OLT,
            5   => CmpFPPredicate::OLE,
            6   => CmpFPPredicate::ONE,
            7   => CmpFPPredicate::ORD,
            8   => CmpFPPredicate::UEQ,
            9   => CmpFPPredicate::UGT,
            10  => CmpFPPredicate::UGE,
            11  => CmpFPPredicate::ULT,
            12  => CmpFPPredicate::ULE,
            13  => CmpFPPredicate::UNE,
            14  => CmpFPPredicate::UNO,
            15  => CmpFPPredicate::AlwaysTrue,
            _   => {
                eprintln!("Invalid value for CmpFPPredicate: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl CmpIPredicate {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => CmpIPredicate::Eq,
            1   => CmpIPredicate::Ne,
            2   => CmpIPredicate::Slt,
            3   => CmpIPredicate::Sle,
            4   => CmpIPredicate::Sgt,
            5   => CmpIPredicate::Sge,
            6   => CmpIPredicate::Ult,
            7   => CmpIPredicate::Ule,
            8   => CmpIPredicate::Ugt,
            9   => CmpIPredicate::Uge,
            _   => {
                eprintln!("Invalid value for CmpIPredicate: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl FastMathFlags {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => FastMathFlags::None,
            1   => FastMathFlags::ReAssoc,
            2   => FastMathFlags::NNaN,
            4   => FastMathFlags::NInf,
            8   => FastMathFlags::NSz,
            16  => FastMathFlags::ARcp,
            32  => FastMathFlags::Contract,
            64  => FastMathFlags::AFn,
            127 => FastMathFlags::Fast,
            _   => {
                eprintln!("Invalid value for FastMathFlags: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl IntegerOverflowFlags {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => IntegerOverflowFlags::None,
            1   => IntegerOverflowFlags::NSW,
            2   => IntegerOverflowFlags::NUW,
            _   => {
                eprintln!("Invalid value for IntegerOverflowFlags: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::AddF            => "addf",
            Op::AddI            => "addi",
            Op::AddUIExtended   => "addui_extended",
            Op::AndI            => "andi",
            Op::Bitcast         => "bitcast",
            Op::CeilDivSI       => "ceildivsi",
            Op::CeilDivUI       => "ceildivui",
            Op::CmpF            => "cmpf",
            Op::CmpI            => "cmpi",
            Op::Constant        => "constant",
            Op::DivF            => "divf",
            Op::DivSI           => "divsi",
            Op::DivUI           => "divui",
            Op::ExtF            => "extf",
            Op::ExtSI           => "extsi",
            Op::ExtUI           => "extui",
            Op::FloorDivSI      => "floordivsi",
            Op::FPToSI          => "fptosi",
            Op::FPToUI          => "fptoui",
            Op::IndexCast       => "index_cast",
            Op::IndexCastUI     => "index_castui",
            Op::MaximumF        => "maximumf",
            Op::MaxNumF         => "maxnumf",
            Op::MaxSI           => "maxsi",
            Op::MaxUI           => "maxui",
            Op::MinimumF        => "minimumf",
            Op::MinNumF         => "minnumf",
            Op::MinSI           => "minsi",
            Op::MinUI           => "minui",
            Op::MulF            => "mulf",
            Op::MulI            => "muli",
            Op::MulSIExtended   => "mulsi_extended",
            Op::MulUIExtended   => "mului_extended",
            Op::NegF            => "negf",
            Op::OrI             => "ori",
            Op::RemF            => "remf",
            Op::RemSI           => "remsi",
            Op::RemUI           => "remui",
            Op::Select          => "select",
            Op::ShlI            => "shli",
            Op::ShrSI           => "shrsi",
            Op::ShrUI           => "shrui",
            Op::SIToFP          => "sitofp",
            Op::SubF            => "subf",
            Op::SubI            => "subi",
            Op::TruncF          => "truncf",
            Op::TruncI          => "trunci",
            Op::UIToFP          => "uitofp",
            Op::XorI            => "xori",
        }
    }
}

impl RoundingMode {
    pub fn from_i64(k: i64) -> Self {
        match k {
            0   => RoundingMode::ToNearestEven,
            1   => RoundingMode::Downward,
            2   => RoundingMode::Upward,
            3   => RoundingMode::TowardZero,
            4   => RoundingMode::ToNearestAway,
            _   => {
                eprintln!("Invalid value for RoundingMode: {}", k);
                exit(ExitCode::DialectError);
            },
        }
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

impl AddF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for AddF operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_float() {
            eprintln!("Expected integer types for AddF operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::AddF.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = FastMath::new(&context, flags);
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get()) 
    }

    pub fn from(op: MlirOperation) -> Self {
        AddF(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl AddI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: IntegerOverflowFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for AddI operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for AddI operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::AddI.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = IntegerOverflow::new(&context, flags);
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        AddI(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        IntegerOverflow::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl AddUIExtended {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for AddUIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for AddUIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::AddUIExtended.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), IntegerType::new_bool(&context).as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        AddUIExtended(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_overflow(&self) -> Value {
        self.as_operation().get_result(1)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl Constant {
    fn new(attr: &Attribute, loc: &Location) -> Self {
        let context = attr.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Constant.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr_name = StringBacked::from_string(&"value".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_float(attr: &FloatAttr, loc: &Location) -> Self {
        Self::new(&attr.as_attribute(), loc)
    }

    pub fn new_integer(attr: &IntegerAttr, loc: &Location) -> Self {
        Self::new(&attr.as_attribute(), loc)
    }

    pub fn from(op: MlirOperation) -> Self {
        Constant(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_float_value(&self) -> Option<FloatAttr> {
        let attr = self.get_value();
        if attr.is_float() {
            Some(FloatAttr::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_integer_value(&self) -> Option<IntegerAttr> {
        let attr = self.get_value();
        if attr.is_integer() {
            Some(IntegerAttr::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_value(&self) -> Attribute {
        let attr_name = StringBacked::from_string(&"value".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Attribute::from(*attr.get())
    }

    pub fn is_float(&self) -> bool {
        self.get_value().is_float()
    }

    pub fn is_integer(&self) -> bool {
        self.get_value().is_integer()
    }
}

impl DivF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for DivF operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_float() {
            eprintln!("Expected integer types for DivF operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::DivF.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = FastMath::new(&context, flags);
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get()) 
    }

    pub fn from(op: MlirOperation) -> Self {
        DivF(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl DivSI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for DivSI operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for DivSI operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::DivSI.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        DivSI(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl DivUI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for DivUI operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for DivUI operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::DivUI.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        DivUI(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl MulF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for MulF operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_float() {
            eprintln!("Expected integer types for MulF operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::MulF.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = FastMath::new(&context, flags);
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get()) 
    }

    pub fn from(op: MlirOperation) -> Self {
        MulF(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl MulI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: IntegerOverflowFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for MulI operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for MulI operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::MulI.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = IntegerOverflow::new(&context, flags);
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        MulI(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        IntegerOverflow::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl MulSIExtended {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for MulSIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for MulSIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::MulSIExtended.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        MulSIExtended(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result_high(&self) -> Value {
        self.as_operation().get_result(1)
    }

    pub fn get_result_low(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl MulUIExtended {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for MulUIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for MulUIExtended operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::MulUIExtended.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        MulUIExtended(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result_high(&self) -> Value {
        self.as_operation().get_result(1)
    }

    pub fn get_result_low(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl SubF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for SubF operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_float() {
            eprintln!("Expected integer types for SubF operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::SubF.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = FastMath::new(&context, flags);
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        SubF(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from_string(&"fastmath".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

impl SubI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: IntegerOverflowFlags, loc: &Location) -> Self {
        if *t != lhs.get_type() || *t != rhs.get_type() {
            eprintln!("Expected matching types for SubI operands and result");
            exit(ExitCode::DialectError);
        }
        if !t.is_integer() {
            eprintln!("Expected integer types for SubI operands and result");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::SubI.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let attr = IntegerOverflow::new(&context, flags);
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&id, &attr.as_attribute());
        op_state.add_attributes(&[attr_named]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        SubI(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from_string(&"overflow".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        IntegerOverflow::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_lhs(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_rhs(&self) -> Value {
        self.as_operation().get_operand(1)
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl DialectOperation for AddF {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::AddF.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::AddF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for AddF {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for AddF {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for AddI {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithIntegerOverflowFlagsInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::AddI.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::AddI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for AddI {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for AddI {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for AddUIExtended {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::AddUIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::AddUIExtended
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for AddUIExtended {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for AddUIExtended {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl DialectOperation for Constant {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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

    fn get_name(&self) -> &'static str {
        Op::Constant.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::Constant
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ConstantLike,
        ]
    }
}

impl Destroy for Constant {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for Constant {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_value() == rhs.get_value()
    }
}

impl DialectOperation for DivF {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::DivF.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::DivF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for DivF {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for DivF {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for DivSI {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::DivSI.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::DivSI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for DivSI {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for DivSI {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl DialectOperation for DivUI {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::DivUI.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::DivUI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for DivUI {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for DivUI {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl IRAttribute for FastMath {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl cmp::PartialEq for FastMath {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

impl IRAttribute for IntegerOverflow {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl cmp::PartialEq for IntegerOverflow {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

impl DialectOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl DialectOperation for MulF {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::MulF.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::MulF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for MulF {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for MulF {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for MulI {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithIntegerOverflowFlagsInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::MulI.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::MulI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for MulI {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for MulI {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for MulSIExtended {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::MulSIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::MulSIExtended
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for MulSIExtended {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for MulSIExtended {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl DialectOperation for MulUIExtended {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::MulUIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::MulUIExtended
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for MulUIExtended {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for MulUIExtended {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl DialectOperation for SubF {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::SubF.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::SubF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for SubF {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for SubF {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

impl DialectOperation for SubI {
    fn as_operation(&self) -> Operation {
        Operation::from(self.0)
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithIntegerOverflowFlagsInterface,
            Interface::ConditionallySpeculatable,
            Interface::InferIntRangeInterface,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_name(&self) -> &'static str {
        Op::SubI.get_name()
    }

    fn get_op(&self) -> &'static dyn DialectOp {
        &Op::SubI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl Destroy for SubI {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
    }
}

impl cmp::PartialEq for SubI {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_flags() == rhs.get_flags()
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for AtomicRMWKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            AtomicRMWKind::AddF     => "addf",
            AtomicRMWKind::AddI     => "addi",
            AtomicRMWKind::Assign   => "assign",
            AtomicRMWKind::MaximumF => "maximumf",
            AtomicRMWKind::MaxS     => "maxs",
            AtomicRMWKind::MaxU     => "maxu",
            AtomicRMWKind::MinimumF => "minumumf",
            AtomicRMWKind::MinS     => "mins",
            AtomicRMWKind::MinU     => "minu",
            AtomicRMWKind::MulF     => "mulf",
            AtomicRMWKind::MulI     => "muli",
            AtomicRMWKind::OrI      => "ori",
            AtomicRMWKind::AndI     => "andi",
            AtomicRMWKind::MaxNumF  => "maxnumf",
            AtomicRMWKind::MinNumF  => "minnumf",
        })
    }
}

impl fmt::Display for CmpFPPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CmpFPPredicate::AlwaysFalse => "false",
            CmpFPPredicate::OEQ         => "oeq",
            CmpFPPredicate::OGT         => "ogt",
            CmpFPPredicate::OGE         => "oge",
            CmpFPPredicate::OLT         => "olt",
            CmpFPPredicate::OLE         => "ole",
            CmpFPPredicate::ONE         => "one",
            CmpFPPredicate::ORD         => "ord",
            CmpFPPredicate::UEQ         => "ueq",
            CmpFPPredicate::UGT         => "ugt",
            CmpFPPredicate::UGE         => "uge",
            CmpFPPredicate::ULT         => "ult",
            CmpFPPredicate::ULE         => "ule",
            CmpFPPredicate::UNE         => "une",
            CmpFPPredicate::UNO         => "uno",
            CmpFPPredicate::AlwaysTrue  => "true",
        })
    }
}

impl fmt::Display for CmpIPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CmpIPredicate::Eq   => "eq",
            CmpIPredicate::Ne   => "ne",
            CmpIPredicate::Slt  => "slt",
            CmpIPredicate::Sle  => "sle",
            CmpIPredicate::Sgt  => "sgt",
            CmpIPredicate::Sge  => "sge",
            CmpIPredicate::Ult  => "ult",
            CmpIPredicate::Ule  => "ule",
            CmpIPredicate::Ugt  => "ugt",
            CmpIPredicate::Uge  => "uge",
        })
    }
}

impl fmt::Display for FastMathFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            FastMathFlags::None     => "none",
            FastMathFlags::ReAssoc  => "reassoc",
            FastMathFlags::NNaN     => "nnan",
            FastMathFlags::NInf     => "ninf",
            FastMathFlags::NSz      => "nsz",
            FastMathFlags::ARcp     => "arcp",
            FastMathFlags::Contract => "contract",
            FastMathFlags::AFn      => "afn",
            FastMathFlags::Fast     => "fast",
        })
    }
}

impl fmt::Display for IntegerOverflowFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            IntegerOverflowFlags::None  => "none",
            IntegerOverflowFlags::NSW   => "nsw",
            IntegerOverflowFlags::NUW   => "nuw",
        })
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::AddF            => "AddFOp",
            Op::AddI            => "AddIOp",
            Op::AddUIExtended   => "AddUIExtendedOp",
            Op::AndI            => "AndIOp",
            Op::Bitcast         => "BitcastOp",
            Op::CeilDivSI       => "CeilDivSIOp",
            Op::CeilDivUI       => "CeilDivUIOp",
            Op::CmpF            => "CmpFOp",
            Op::CmpI            => "CmpIOp",
            Op::Constant        => "ConstantOp",
            Op::DivF            => "DivFOp",
            Op::DivSI           => "DivSIOp",
            Op::DivUI           => "DivUIOp",
            Op::ExtF            => "ExtFOp",
            Op::ExtSI           => "ExtSIOp",
            Op::ExtUI           => "ExtUIOp",
            Op::FloorDivSI      => "FloorDivSIOp",
            Op::FPToSI          => "FPToSIOp",
            Op::FPToUI          => "FPToUIOp",
            Op::IndexCast       => "IndexCastOp",
            Op::IndexCastUI     => "IndexCastUIOp",
            Op::MaximumF        => "MaximumFOp",
            Op::MaxNumF         => "MaxNumFOp",
            Op::MaxSI           => "MaxSIOp",
            Op::MaxUI           => "MaxUIOp",
            Op::MinimumF        => "MinimumFOp",
            Op::MinNumF         => "MinNumFOp",
            Op::MinSI           => "MinSIOp",
            Op::MinUI           => "MinUIOp",
            Op::MulF            => "MulFOp",
            Op::MulI            => "MulIOp",
            Op::MulSIExtended   => "MulSIExtendedOp",
            Op::MulUIExtended   => "MulUIExtendedOp",
            Op::NegF            => "NegFOp",
            Op::OrI             => "OrIOp",
            Op::RemF            => "RemFOp",
            Op::RemSI           => "RemSIOp",
            Op::RemUI           => "RemUIOp",
            Op::Select          => "SelectOp",
            Op::ShlI            => "ShlIOp",
            Op::ShrSI           => "ShrSIOp",
            Op::ShrUI           => "ShrUIOp",
            Op::SIToFP          => "SIToFPOp",
            Op::SubF            => "SubFOp",
            Op::SubI            => "SubIOp",
            Op::TruncF          => "TruncFOp",
            Op::TruncI          => "TruncIOp",
            Op::UIToFP          => "UIToFPOp",
            Op::XorI            => "XorIOp",
        })
    }
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            RoundingMode::ToNearestEven => "to_nearest_even",
            RoundingMode::Downward      => "downward",
            RoundingMode::Upward        => "upward",
            RoundingMode::TowardZero    => "toward_zero",
            RoundingMode::ToNearestAway => "to_nearest_away",
        })
    }
}
