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

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::float::Float as FloatAttr;
use attributes::index::Index as IndexAttr;
use attributes::integer::Integer as IntegerAttr;
use attributes::named::Named;
use attributes::specialized::CustomAttributeData;
use attributes::specialized::NamedFloatOrIndexOrInteger;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedParsed;
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
use ir::Shape;
use ir::ShapeImpl;
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::GetWidth;
use types::IType;
use types::integer::Integer as IntegerType;
use types::ranked_tensor::RankedTensor;
use types::shaped::Shaped;
use types::vector::Vector;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct ArithValue(MlirAttribute);

#[derive(Clone)]
pub struct FastMath(MlirAttribute);

#[derive(Clone)]
pub struct IntegerOverflow(MlirAttribute);

#[derive(Clone)]
pub struct RoundingMode(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum AtomicRMWKind {
    AddF = 0,
    AddI = 1,
    Assign = 2,
    MaximumF = 3,
    MaxS = 4,
    MaxU = 5,
    MinimumF = 6,
    MinS = 7,
    MinU = 8,
    MulF = 9,
    MulI = 10,
    OrI = 11,
    AndI = 12,
    MaxNumF = 13,
    MinNumF = 14,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum CmpFPPredicate {
    AlwaysFalse = 0,
    OEQ = 1,
    OGT = 2,
    OGE = 3,
    OLT = 4,
    OLE = 5,
    ONE = 6,
    ORD = 7,
    UEQ = 8,
    UGT = 9,
    UGE = 10,
    ULT = 11,
    ULE = 12,
    UNE = 13,
    UNO = 14,
    AlwaysTrue = 15,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum CmpIPredicate {
    Eq = 0,
    Ne = 1,
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
#[derive(Clone, Copy, PartialEq)]
pub enum FastMathFlags {
    None = 0,
    ReAssoc = 1,
    NNaN = 2,
    NInf = 4,
    NSz = 8,
    ARcp = 16,
    Contract = 32,
    AFn = 64,
    Fast = 127,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum IntegerOverflowFlags {
    None = 0,
    NSW = 1,
    NUW = 2,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum RoundingModeKind {
    ToNearestEven = 0,
    Downward = 1,
    Upward = 2,
    TowardZero = 3,
    ToNearestAway = 4,
}

#[derive(Clone, Copy, PartialEq)]
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
    XOrI,
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
pub struct AndI(MlirOperation);

#[derive(Clone)]
pub struct Bitcast(MlirOperation);

#[derive(Clone)]
pub struct Constant(MlirOperation);

#[derive(Clone)]
pub struct DivF(MlirOperation);

#[derive(Clone)]
pub struct DivSI(MlirOperation);

#[derive(Clone)]
pub struct DivUI(MlirOperation);

#[derive(Clone)]
pub struct ExtF(MlirOperation);

#[derive(Clone)]
pub struct ExtSI(MlirOperation);

#[derive(Clone)]
pub struct ExtUI(MlirOperation);

#[derive(Clone)]
pub struct FPToSI(MlirOperation);

#[derive(Clone)]
pub struct FPToUI(MlirOperation);

#[derive(Clone)]
pub struct IndexCast(MlirOperation);

#[derive(Clone)]
pub struct IndexCastUI(MlirOperation);

#[derive(Clone)]
pub struct MulF(MlirOperation);

#[derive(Clone)]
pub struct MulI(MlirOperation);

#[derive(Clone)]
pub struct MulSIExtended(MlirOperation);

#[derive(Clone)]
pub struct MulUIExtended(MlirOperation);

#[derive(Clone)]
pub struct NegF(MlirOperation);

#[derive(Clone)]
pub struct OrI(MlirOperation);

#[derive(Clone)]
pub struct SIToFP(MlirOperation);

#[derive(Clone)]
pub struct SubF(MlirOperation);

#[derive(Clone)]
pub struct SubI(MlirOperation);

#[derive(Clone)]
pub struct TruncF(MlirOperation);

#[derive(Clone)]
pub struct TruncI(MlirOperation);

#[derive(Clone)]
pub struct UIToFP(MlirOperation);

#[derive(Clone)]
pub struct XOrI(MlirOperation);

///////////////////////////////
//  Support
///////////////////////////////

fn check_binary_operation_types(
    is_type: fn(&Type) -> bool,
    op: Op,
    t: &Type,
    lhs: &Value,
    rhs: &Value,
) -> () {
    let t_lhs = lhs.get_type();
    let t_rhs = rhs.get_type();
    let (t_elem, t_elem_lhs, t_elem_rhs) = if t.is_shaped() {
        let isnt_tensor_vector_result = !t.is_tensor() && !t.is_vector();
        let isnt_tensor_vector_lhs = !t_lhs.is_tensor() && !t_lhs.is_vector();
        let isnt_tensor_vector_rhs = !t_rhs.is_tensor() && !t_rhs.is_vector();
        if isnt_tensor_vector_result || isnt_tensor_vector_lhs || isnt_tensor_vector_rhs {
            eprintln!(
                "Expected tensor or vector types for shaped {} operands and result",
                op.get_name(),
            );
            exit(ExitCode::DialectError);
        }
        let t_elem = Shaped::from(t).get_element_type();
        let t_elem_lhs = Shaped::from(t_lhs).get_element_type();
        let t_elem_rhs = Shaped::from(t_rhs).get_element_type();
        (t_elem, t_elem_lhs, t_elem_rhs)
    } else {
        (t.clone(), t_lhs, t_rhs)
    };
    if !is_type(&t_elem) {
        eprintln!("Unexpected type for {} operands and result", op.get_name());
        exit(ExitCode::DialectError);
    }
    if t_elem != t_elem_lhs || t_elem != t_elem_rhs {
        eprintln!(
            "Expected matching types for {} operands and result",
            op.get_name()
        );
        exit(ExitCode::DialectError);
    }
}

fn check_binary_operation_float_types(op: Op, t: &Type, lhs: &Value, rhs: &Value) -> () {
    let is_type: fn(&Type) -> bool = Type::is_float;
    check_binary_operation_types(is_type, op, t, lhs, rhs);
}

fn check_binary_operation_integer_types(op: Op, t: &Type, lhs: &Value, rhs: &Value) -> () {
    let is_type: fn(&Type) -> bool = |t: &Type| t.is_index() || t.is_integer();
    check_binary_operation_types(is_type, op, t, lhs, rhs);
}

/// TODO: Check memref.
fn check_element_type(is_type: fn(&Type) -> bool, op: Op, t: &Type, do_exit: bool) -> bool {
    let name = op.get_name();
    let t_elem = if t.is_shaped() {
        if !t.is_tensor() && !t.is_vector() {
            if do_exit {
                eprintln!("Expected tensor or vector type for {} operation", name);
                exit(ExitCode::DialectError);
            } else {
                return false;
            }
        }
        Shaped::from(t).get_element_type()
    } else {
        t.clone()
    };
    if is_type(&t_elem) {
        true
    } else if do_exit {
        eprintln!("Unexpected element type for {} operation", name);
        exit(ExitCode::DialectError);
    } else {
        false
    }
}

fn check_element_type_float(op: Op, t: &Type, do_exit: bool) -> bool {
    check_element_type(Type::is_float, op, t, do_exit)
}

fn check_element_type_index(op: Op, t: &Type, do_exit: bool) -> bool {
    check_element_type(Type::is_index, op, t, do_exit)
}

fn check_element_type_integer(op: Op, t: &Type, do_exit: bool) -> bool {
    check_element_type(Type::is_integer, op, t, do_exit)
}

fn check_element_type_integer_like(op: Op, t: &Type, do_exit: bool) -> bool {
    let is_type: fn(&Type) -> bool = |t: &Type| t.is_index() || t.is_integer();
    check_element_type(is_type, op, t, do_exit)
}

fn check_type_shape(op: Op, t_src: &Type, t_dst: &Type) -> () {
    let is_match_memref = t_src.is_memref() && t_dst.is_memref();
    let is_match_tensor = t_src.is_tensor() && t_dst.is_tensor();
    let is_match_vector = t_src.is_vector() && t_dst.is_vector();
    let is_match_non_shaped = !t_src.is_shaped() && !t_dst.is_shaped();
    if !(is_match_memref || is_match_tensor || is_match_vector || is_match_non_shaped) {
        eprintln!(
            "Expected matching shape for source and result type of {} operation",
            op.get_name()
        );
        exit(ExitCode::DialectError);
    }
}

fn check_type_width(op: Op, pred: CmpIPredicate, t_src: &Type, t_dst: &Type) -> () {
    let name = op.get_name();
    let Some(w_src) = t_src.get_width() else {
        eprintln!("Expected width for source type of {} operation", name);
        exit(ExitCode::DialectError);
    };
    let Some(w_dst) = t_dst.get_width() else {
        eprintln!("Expected width for destination type of {} operation", name);
        exit(ExitCode::DialectError);
    };
    if !match pred {
        CmpIPredicate::Eq => w_src == w_dst,
        CmpIPredicate::Ne => w_src != w_dst,
        CmpIPredicate::Slt | CmpIPredicate::Ult => w_src < w_dst,
        CmpIPredicate::Sle | CmpIPredicate::Ule => w_src <= w_dst,
        CmpIPredicate::Sgt | CmpIPredicate::Ugt => w_src > w_dst,
        CmpIPredicate::Sge | CmpIPredicate::Uge => w_src >= w_dst,
    } {
        eprintln!(
            "Expected source type width ({}) {} destination type width ({}) for {} operation",
            w_src,
            pred.get_operator_symbol(),
            w_dst,
            name,
        );
        exit(ExitCode::DialectError);
    }
}

fn check_unary_operation_types(is_type: fn(&Type) -> bool, op: Op, t: &Type, input: &Value) -> () {
    let t_input = input.get_type();
    let (t_elem, t_elem_input) = if t.is_shaped() {
        let isnt_tensor_vector_result = !t.is_tensor() && !t.is_vector();
        let isnt_tensor_vector_input = !t_input.is_tensor() && !t_input.is_vector();
        if isnt_tensor_vector_result || isnt_tensor_vector_input {
            eprintln!(
                "Expected tensor or vector types for shaped {} operands and result",
                op.get_name(),
            );
            exit(ExitCode::DialectError);
        }
        let t_elem = Shaped::from(t).get_element_type();
        let t_elem_input = Shaped::from(t_input).get_element_type();
        (t_elem, t_elem_input)
    } else {
        (t.clone(), t_input)
    };
    if !is_type(&t_elem) {
        eprintln!("Unexpected type for {} operands and result", op.get_name());
        exit(ExitCode::DialectError);
    }
    if t_elem != t_elem_input {
        eprintln!(
            "Expected matching types for {} operands and result",
            op.get_name()
        );
        exit(ExitCode::DialectError);
    }
}

fn check_unary_operation_float_types(op: Op, t: &Type, input: &Value) -> () {
    let is_type: fn(&Type) -> bool = Type::is_float;
    check_unary_operation_types(is_type, op, t, input);
}

fn check_unary_operation_integer_types(op: Op, t: &Type, input: &Value) -> () {
    let is_type: fn(&Type) -> bool = |t: &Type| t.is_index() || t.is_integer();
    check_unary_operation_types(is_type, op, t, input);
}

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl ArithValue {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl FastMath {
    pub fn new(context: &Context, flags: FastMathFlags) -> Self {
        let cad = CustomAttributeData::new(
            Self::get_name().to_string(),
            context.get_dialect_arith().get_namespace().to_string(),
            vec![flags.get_name().to_string()],
        );
        <Self as NamedParsed>::new_custom(context, &cad)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IntegerOverflow {
    pub fn new(context: &Context, flags: IntegerOverflowFlags) -> Self {
        let cad = CustomAttributeData::new(
            "overflow".to_string(),
            context.get_dialect_arith().get_namespace().to_string(),
            vec![flags.get_name().to_string()],
        );
        <Self as NamedParsed>::new_custom(context, &cad)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl RoundingMode {
    pub fn new(context: &Context, k: RoundingModeKind) -> Self {
        const WIDTH: usize = 32;
        Self::from(*<Self as NamedInteger>::new(context, k as i64, WIDTH).get_mut())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_kind(&self) -> RoundingModeKind {
        RoundingModeKind::from(self.get_value())
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

///////////////////////////////
//  Enum Implementation
///////////////////////////////

impl AtomicRMWKind {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => AtomicRMWKind::AddF,
            1 => AtomicRMWKind::AddI,
            2 => AtomicRMWKind::Assign,
            3 => AtomicRMWKind::MaximumF,
            4 => AtomicRMWKind::MaxS,
            5 => AtomicRMWKind::MaxU,
            6 => AtomicRMWKind::MinimumF,
            7 => AtomicRMWKind::MinS,
            8 => AtomicRMWKind::MinU,
            9 => AtomicRMWKind::MulF,
            10 => AtomicRMWKind::MulI,
            11 => AtomicRMWKind::OrI,
            12 => AtomicRMWKind::AndI,
            13 => AtomicRMWKind::MaxNumF,
            14 => AtomicRMWKind::MinNumF,
            _ => {
                eprintln!("Invalid value for AtomicRMWKind: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }
}

impl CmpFPPredicate {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => CmpFPPredicate::AlwaysFalse,
            1 => CmpFPPredicate::OEQ,
            2 => CmpFPPredicate::OGT,
            3 => CmpFPPredicate::OGE,
            4 => CmpFPPredicate::OLT,
            5 => CmpFPPredicate::OLE,
            6 => CmpFPPredicate::ONE,
            7 => CmpFPPredicate::ORD,
            8 => CmpFPPredicate::UEQ,
            9 => CmpFPPredicate::UGT,
            10 => CmpFPPredicate::UGE,
            11 => CmpFPPredicate::ULT,
            12 => CmpFPPredicate::ULE,
            13 => CmpFPPredicate::UNE,
            14 => CmpFPPredicate::UNO,
            15 => CmpFPPredicate::AlwaysTrue,
            _ => {
                eprintln!("Invalid value for CmpFPPredicate: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }
}

impl CmpIPredicate {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => CmpIPredicate::Eq,
            1 => CmpIPredicate::Ne,
            2 => CmpIPredicate::Slt,
            3 => CmpIPredicate::Sle,
            4 => CmpIPredicate::Sgt,
            5 => CmpIPredicate::Sge,
            6 => CmpIPredicate::Ult,
            7 => CmpIPredicate::Ule,
            8 => CmpIPredicate::Ugt,
            9 => CmpIPredicate::Uge,
            _ => {
                eprintln!("Invalid value for CmpIPredicate: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }

    pub fn get_operator_symbol(&self) -> &'static str {
        match self {
            CmpIPredicate::Eq => "==",
            CmpIPredicate::Ne => "!=",
            CmpIPredicate::Sge | CmpIPredicate::Uge => ">=",
            CmpIPredicate::Sgt | CmpIPredicate::Ugt => ">",
            CmpIPredicate::Sle | CmpIPredicate::Ule => "<=",
            CmpIPredicate::Slt | CmpIPredicate::Ult => "<",
        }
    }
}

impl FastMathFlags {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => FastMathFlags::None,
            1 => FastMathFlags::ReAssoc,
            2 => FastMathFlags::NNaN,
            4 => FastMathFlags::NInf,
            8 => FastMathFlags::NSz,
            16 => FastMathFlags::ARcp,
            32 => FastMathFlags::Contract,
            64 => FastMathFlags::AFn,
            127 => FastMathFlags::Fast,
            _ => {
                eprintln!("Invalid value for FastMathFlags: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            FastMathFlags::None => "none",
            FastMathFlags::ReAssoc => "reassoc",
            FastMathFlags::NNaN => "nnan",
            FastMathFlags::NInf => "ninf",
            FastMathFlags::NSz => "nsz",
            FastMathFlags::ARcp => "arcp",
            FastMathFlags::Contract => "contract",
            FastMathFlags::AFn => "afn",
            FastMathFlags::Fast => "fast",
        }
    }
}

impl IntegerOverflowFlags {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => IntegerOverflowFlags::None,
            1 => IntegerOverflowFlags::NSW,
            2 => IntegerOverflowFlags::NUW,
            _ => {
                eprintln!("Invalid value for IntegerOverflowFlags: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            IntegerOverflowFlags::None => "none",
            IntegerOverflowFlags::NSW => "nsw",
            IntegerOverflowFlags::NUW => "nuw",
        }
    }
}

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::AddF => "addf",
            Op::AddI => "addi",
            Op::AddUIExtended => "addui_extended",
            Op::AndI => "andi",
            Op::Bitcast => "bitcast",
            Op::CeilDivSI => "ceildivsi",
            Op::CeilDivUI => "ceildivui",
            Op::CmpF => "cmpf",
            Op::CmpI => "cmpi",
            Op::Constant => "constant",
            Op::DivF => "divf",
            Op::DivSI => "divsi",
            Op::DivUI => "divui",
            Op::ExtF => "extf",
            Op::ExtSI => "extsi",
            Op::ExtUI => "extui",
            Op::FloorDivSI => "floordivsi",
            Op::FPToSI => "fptosi",
            Op::FPToUI => "fptoui",
            Op::IndexCast => "index_cast",
            Op::IndexCastUI => "index_castui",
            Op::MaximumF => "maximumf",
            Op::MaxNumF => "maxnumf",
            Op::MaxSI => "maxsi",
            Op::MaxUI => "maxui",
            Op::MinimumF => "minimumf",
            Op::MinNumF => "minnumf",
            Op::MinSI => "minsi",
            Op::MinUI => "minui",
            Op::MulF => "mulf",
            Op::MulI => "muli",
            Op::MulSIExtended => "mulsi_extended",
            Op::MulUIExtended => "mului_extended",
            Op::NegF => "negf",
            Op::OrI => "ori",
            Op::RemF => "remf",
            Op::RemSI => "remsi",
            Op::RemUI => "remui",
            Op::Select => "select",
            Op::ShlI => "shli",
            Op::ShrSI => "shrsi",
            Op::ShrUI => "shrui",
            Op::SIToFP => "sitofp",
            Op::SubF => "subf",
            Op::SubI => "subi",
            Op::TruncF => "truncf",
            Op::TruncI => "trunci",
            Op::UIToFP => "uitofp",
            Op::XOrI => "xori",
        }
    }
}

impl RoundingModeKind {
    pub fn from_i32(k: i32) -> Self {
        match k {
            0 => RoundingModeKind::ToNearestEven,
            1 => RoundingModeKind::Downward,
            2 => RoundingModeKind::Upward,
            3 => RoundingModeKind::TowardZero,
            4 => RoundingModeKind::ToNearestAway,
            _ => {
                eprintln!("Invalid value for RoundingMode: {}", k);
                exit(ExitCode::DialectError);
            }
        }
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

impl AddF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        check_binary_operation_float_types(Op::AddF, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::AddF);
        let attr = FastMath::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
    pub fn new(
        t: &Type,
        lhs: &Value,
        rhs: &Value,
        flags: IntegerOverflowFlags,
        loc: &Location,
    ) -> Self {
        check_binary_operation_integer_types(Op::AddI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::AddI);
        let attr = IntegerOverflow::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from(IntegerOverflow::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
        check_binary_operation_integer_types(Op::AddUIExtended, t, lhs, rhs);
        let context = t.get_context();
        let s = if t.is_shaped() {
            Some(ShapeImpl::from(Shaped::from(t).to_vec()))
        } else {
            None
        };
        let t_flags_elem = IntegerType::new_signless(&context, 1).as_type();
        let t_flags = if t.is_tensor() {
            RankedTensor::new(&s.unwrap(), &t_flags_elem).as_type()
        } else if t.is_vector() {
            Vector::new(&s.unwrap(), &t_flags_elem).as_type()
        } else {
            t_flags_elem
        };
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::AddUIExtended);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), t_flags]);
        Self::from(*op_state.create_operation().get())
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

impl AndI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        check_binary_operation_integer_types(Op::AndI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::AndI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
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

impl Bitcast {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_type_shape(Op::Bitcast, &t_input, t);
        check_type_width(Op::Bitcast, CmpIPredicate::Eq, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::Bitcast);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
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
    fn new(t: &Type, attr: &ArithValue, loc: &Location) -> Self {
        if !t.is_float() && !t.is_index() && !t.is_integer() {
            eprintln!("Expected float, index, or integer arith value for constant");
            exit(ExitCode::DialectError);
        }
        let context = attr.as_attribute().get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::Constant);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_float(attr: &FloatAttr, loc: &Location) -> Self {
        let t = attr.get_type();
        Self::new(&t, &ArithValue::new_float(attr), loc)
    }

    pub fn new_index(attr: &IndexAttr, loc: &Location) -> Self {
        let t = attr.get_type();
        Self::new(&t, &ArithValue::new_index(attr), loc)
    }

    pub fn new_integer(attr: &IntegerAttr, loc: &Location) -> Self {
        let t = attr.get_type();
        Self::new(&t, &ArithValue::new_integer(attr), loc)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_float_value(&self) -> Option<FloatAttr> {
        self.get_value().as_float()
    }

    pub fn get_integer_value(&self) -> Option<IntegerAttr> {
        self.get_value().as_integer()
    }

    pub fn get_value(&self) -> ArithValue {
        let attr_name = StringBacked::from(ArithValue::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        ArithValue::from(*attr.get())
    }

    pub fn is_float(&self) -> bool {
        self.get_value().is_float()
    }

    pub fn is_index(&self) -> bool {
        self.get_value().is_index()
    }

    pub fn is_integer(&self) -> bool {
        self.get_value().is_integer()
    }
}

impl DivF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        check_binary_operation_float_types(Op::DivF, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::DivF);
        let attr = FastMath::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
        check_binary_operation_integer_types(Op::DivSI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::DivSI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
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
        check_binary_operation_integer_types(Op::DivUI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::DivUI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
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

impl ExtF {
    pub fn new(t: &Type, input: &Value, flags: Option<FastMathFlags>, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_float(Op::ExtF, t, true);
        check_element_type_float(Op::ExtF, &t_input, true);
        check_type_shape(Op::ExtF, &t_input, t);
        check_type_width(Op::ExtF, CmpIPredicate::Slt, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::ExtF);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        if let Some(flags_) = flags {
            let attr_fastmath = FastMath::new(&context, flags_);
            op_state.add_attributes(&[attr_fastmath.as_named_attribute()]);
        }
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl ExtSI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::ExtSI, t, true);
        check_element_type_integer_like(Op::ExtSI, &t_input, true);
        check_type_shape(Op::ExtSI, &t_input, t);
        check_type_width(Op::ExtSI, CmpIPredicate::Slt, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::ExtSI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl ExtUI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::ExtUI, t, true);
        check_element_type_integer_like(Op::ExtUI, &t_input, true);
        check_type_shape(Op::ExtUI, &t_input, t);
        check_type_width(Op::ExtUI, CmpIPredicate::Slt, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::ExtUI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl FPToSI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::FPToSI, t, true);
        check_element_type_float(Op::FPToSI, &t_input, true);
        check_type_shape(Op::FPToSI, &t_input, t);
        check_type_width(Op::FPToSI, CmpIPredicate::Eq, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::FPToSI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl FPToUI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::FPToUI, t, true);
        check_element_type_float(Op::FPToUI, &t_input, true);
        check_type_shape(Op::FPToUI, &t_input, t);
        check_type_width(Op::FPToUI, CmpIPredicate::Eq, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::FPToUI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl IndexCast {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        if check_element_type_index(Op::IndexCast, t, false) {
            check_element_type_integer(Op::IndexCast, &t_input, true);
        } else {
            check_element_type_index(Op::IndexCast, &t_input, true);
        }
        check_type_shape(Op::IndexCast, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::IndexCast);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl IndexCastUI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        if check_element_type_index(Op::IndexCast, t, false) {
            check_element_type_integer(Op::IndexCast, &t_input, true);
        } else {
            check_element_type_index(Op::IndexCast, &t_input, true);
        }
        check_type_shape(Op::IndexCastUI, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::IndexCastUI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl MulF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        check_binary_operation_float_types(Op::MulF, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::MulF);
        let attr = FastMath::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
    pub fn new(
        t: &Type,
        lhs: &Value,
        rhs: &Value,
        flags: IntegerOverflowFlags,
        loc: &Location,
    ) -> Self {
        check_binary_operation_integer_types(Op::MulI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::MulI);
        let attr = IntegerOverflow::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from(IntegerOverflow::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
        check_binary_operation_integer_types(Op::MulSIExtended, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::MulSIExtended);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), t.clone()]);
        Self::from(*op_state.create_operation().get())
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
        check_binary_operation_integer_types(Op::MulUIExtended, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::MulUIExtended);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone(), t.clone()]);
        Self::from(*op_state.create_operation().get())
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

impl NegF {
    pub fn new(t: &Type, input: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        check_unary_operation_float_types(Op::NegF, t, input);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::NegF);
        let attr = FastMath::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl OrI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        check_binary_operation_integer_types(Op::OrI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::OrI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
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

impl SIToFP {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::SIToFP, &t_input, true);
        check_element_type_float(Op::SIToFP, t, true);
        check_type_shape(Op::SIToFP, &t_input, t);
        check_type_width(Op::SIToFP, CmpIPredicate::Eq, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::SIToFP);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl SubF {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, flags: FastMathFlags, loc: &Location) -> Self {
        check_binary_operation_float_types(Op::SubF, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::SubF);
        let attr = FastMath::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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
    pub fn new(
        t: &Type,
        lhs: &Value,
        rhs: &Value,
        flags: IntegerOverflowFlags,
        loc: &Location,
    ) -> Self {
        check_binary_operation_integer_types(Op::SubI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::SubI);
        let attr = IntegerOverflow::new(&context, flags);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> IntegerOverflow {
        let attr_name = StringBacked::from(IntegerOverflow::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
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

impl TruncF {
    pub fn new(
        t: &Type,
        input: &Value,
        flags: Option<FastMathFlags>,
        mode: Option<RoundingModeKind>,
        loc: &Location,
    ) -> Self {
        let t_input = input.get_type();
        check_element_type_float(Op::TruncF, t, true);
        check_element_type_float(Op::TruncF, &t_input, true);
        check_type_shape(Op::TruncF, &t_input, t);
        check_type_width(Op::TruncF, CmpIPredicate::Sgt, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::TruncF);
        let mut attrs: Vec<Named> = vec![];
        if let Some(flags_) = flags {
            let attr_fastmath = FastMath::new(&context, flags_);
            attrs.push(attr_fastmath.as_named_attribute())
        }
        if let Some(mode_) = mode {
            let attr_rounding_mode = RoundingMode::new(&context, mode_);
            attrs.push(attr_rounding_mode.as_named_attribute());
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&attrs);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_flags(&self) -> FastMath {
        let attr_name = StringBacked::from(FastMath::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        FastMath::from(*attr.get())
    }

    pub fn get_mode(&self) -> RoundingMode {
        let attr_name = StringBacked::from(RoundingMode::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        RoundingMode::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl TruncI {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::TruncI, t, true);
        check_element_type_integer_like(Op::TruncI, &t_input, true);
        check_type_shape(Op::TruncI, &t_input, t);
        check_type_width(Op::TruncI, CmpIPredicate::Sgt, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::TruncI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl UIToFP {
    pub fn new(t: &Type, input: &Value, loc: &Location) -> Self {
        let t_input = input.get_type();
        check_element_type_integer_like(Op::UIToFP, &t_input, true);
        check_element_type_float(Op::UIToFP, t, true);
        check_type_shape(Op::UIToFP, &t_input, t);
        check_type_width(Op::UIToFP, CmpIPredicate::Eq, &t_input, t);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::UIToFP);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[input.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl XOrI {
    pub fn new(t: &Type, lhs: &Value, rhs: &Value, loc: &Location) -> Self {
        check_binary_operation_integer_types(Op::XOrI, t, lhs, rhs);
        let context = t.get_context();
        let dialect = context.get_dialect_arith();
        let name = dialect.get_op_name(&Op::XOrI);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[lhs.clone(), rhs.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
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

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl From<MlirOperation> for AddF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for AddF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::AddF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::AddF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for AddI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for AddI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::AddI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::AddI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for AddUIExtended {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for AddUIExtended {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::AddUIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
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

impl From<MlirOperation> for AndI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for AndI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::AndI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::AndI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::Idempotent,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

SpecializedAttribute!("value" = impl NamedFloatOrIndexOrInteger for ArithValue {});

impl From<i32> for AtomicRMWKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for AtomicRMWKind {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl From<MlirOperation> for Bitcast {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Bitcast {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Bitcast.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Bitcast
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<i32> for CmpFPPredicate {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for CmpFPPredicate {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl From<i32> for CmpIPredicate {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for CmpIPredicate {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
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
        self.as_operation().get_context().get_dialect_arith()
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

impl From<MlirOperation> for DivF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::DivF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for DivSI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivSI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivSI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::DivSI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for DivUI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivUI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivUI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::DivUI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for ExtF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ExtF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ExtF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::ExtF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for ExtSI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ExtSI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ExtSI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::ExtSI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for ExtUI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ExtUI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ExtUI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::ExtUI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

SpecializedAttribute!("fastmath" = impl NamedParsed for FastMath {});

impl From<i32> for FastMathFlags {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for FastMathFlags {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl From<MlirOperation> for FPToSI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for FPToSI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::FPToSI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::FPToSI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for FPToUI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for FPToUI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::FPToUI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::FPToUI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for IndexCast {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for IndexCast {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::IndexCast.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::IndexCast
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for IndexCastUI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for IndexCastUI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::IndexCastUI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::IndexCastUI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

SpecializedAttribute!("overflowFlags" = impl NamedParsed for IntegerOverflow {});

impl From<i32> for IntegerOverflowFlags {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for IntegerOverflowFlags {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for MulF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MulF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MulF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::MulF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for MulI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MulI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MulI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::MulI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for MulSIExtended {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MulSIExtended {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MulSIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
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

impl From<MlirOperation> for MulUIExtended {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MulUIExtended {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MulUIExtended.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
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

impl From<MlirOperation> for NegF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for NegF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::NegF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::NegF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for OrI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for OrI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::OrI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::OrI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::Idempotent,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

SpecializedAttribute!("roundingmode" = impl NamedInteger for RoundingMode {});

impl From<i32> for RoundingModeKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

impl From<i64> for RoundingModeKind {
    fn from(n: i64) -> Self {
        Self::from(n as i32)
    }
}

impl From<MlirOperation> for SIToFP {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for SIToFP {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::SIToFP.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::SIToFP
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for SubF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for SubF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::SubF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::SubF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for SubI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for SubI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::SubI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::SubI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for TruncF {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for TruncF {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ArithFastMathInterface,
            Interface::ArithRoundingModeInterface,
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::TruncF.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::TruncF
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for TruncI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for TruncI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::TruncI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::TruncI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for UIToFP {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for UIToFP {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::UIToFP.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::UIToFP
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

impl From<MlirOperation> for XOrI {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for XOrI {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_arith()
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
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::XOrI.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::XOrI
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::Commutative,
            Trait::ElementWise,
            Trait::SameOperandsAndResultType,
            Trait::Scalarizable,
            Trait::Tensorizable,
            Trait::Vectorizable,
        ]
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for AtomicRMWKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            AtomicRMWKind::AddF => "addf",
            AtomicRMWKind::AddI => "addi",
            AtomicRMWKind::Assign => "assign",
            AtomicRMWKind::MaximumF => "maximumf",
            AtomicRMWKind::MaxS => "maxs",
            AtomicRMWKind::MaxU => "maxu",
            AtomicRMWKind::MinimumF => "minumumf",
            AtomicRMWKind::MinS => "mins",
            AtomicRMWKind::MinU => "minu",
            AtomicRMWKind::MulF => "mulf",
            AtomicRMWKind::MulI => "muli",
            AtomicRMWKind::OrI => "ori",
            AtomicRMWKind::AndI => "andi",
            AtomicRMWKind::MaxNumF => "maxnumf",
            AtomicRMWKind::MinNumF => "minnumf",
        })
    }
}

impl fmt::Display for CmpFPPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CmpFPPredicate::AlwaysFalse => "false",
            CmpFPPredicate::OEQ => "oeq",
            CmpFPPredicate::OGT => "ogt",
            CmpFPPredicate::OGE => "oge",
            CmpFPPredicate::OLT => "olt",
            CmpFPPredicate::OLE => "ole",
            CmpFPPredicate::ONE => "one",
            CmpFPPredicate::ORD => "ord",
            CmpFPPredicate::UEQ => "ueq",
            CmpFPPredicate::UGT => "ugt",
            CmpFPPredicate::UGE => "uge",
            CmpFPPredicate::ULT => "ult",
            CmpFPPredicate::ULE => "ule",
            CmpFPPredicate::UNE => "une",
            CmpFPPredicate::UNO => "uno",
            CmpFPPredicate::AlwaysTrue => "true",
        })
    }
}

impl fmt::Display for CmpIPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CmpIPredicate::Eq => "eq",
            CmpIPredicate::Ne => "ne",
            CmpIPredicate::Slt => "slt",
            CmpIPredicate::Sle => "sle",
            CmpIPredicate::Sgt => "sgt",
            CmpIPredicate::Sge => "sge",
            CmpIPredicate::Ult => "ult",
            CmpIPredicate::Ule => "ule",
            CmpIPredicate::Ugt => "ugt",
            CmpIPredicate::Uge => "uge",
        })
    }
}

impl fmt::Display for FastMathFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_name())
    }
}

impl fmt::Display for IntegerOverflowFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get_name())
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::AddF => "AddFOp",
            Op::AddI => "AddIOp",
            Op::AddUIExtended => "AddUIExtendedOp",
            Op::AndI => "AndIOp",
            Op::Bitcast => "BitcastOp",
            Op::CeilDivSI => "CeilDivSIOp",
            Op::CeilDivUI => "CeilDivUIOp",
            Op::CmpF => "CmpFOp",
            Op::CmpI => "CmpIOp",
            Op::Constant => "ConstantOp",
            Op::DivF => "DivFOp",
            Op::DivSI => "DivSIOp",
            Op::DivUI => "DivUIOp",
            Op::ExtF => "ExtFOp",
            Op::ExtSI => "ExtSIOp",
            Op::ExtUI => "ExtUIOp",
            Op::FloorDivSI => "FloorDivSIOp",
            Op::FPToSI => "FPToSIOp",
            Op::FPToUI => "FPToUIOp",
            Op::IndexCast => "IndexCastOp",
            Op::IndexCastUI => "IndexCastUIOp",
            Op::MaximumF => "MaximumFOp",
            Op::MaxNumF => "MaxNumFOp",
            Op::MaxSI => "MaxSIOp",
            Op::MaxUI => "MaxUIOp",
            Op::MinimumF => "MinimumFOp",
            Op::MinNumF => "MinNumFOp",
            Op::MinSI => "MinSIOp",
            Op::MinUI => "MinUIOp",
            Op::MulF => "MulFOp",
            Op::MulI => "MulIOp",
            Op::MulSIExtended => "MulSIExtendedOp",
            Op::MulUIExtended => "MulUIExtendedOp",
            Op::NegF => "NegFOp",
            Op::OrI => "OrIOp",
            Op::RemF => "RemFOp",
            Op::RemSI => "RemSIOp",
            Op::RemUI => "RemUIOp",
            Op::Select => "SelectOp",
            Op::ShlI => "ShlIOp",
            Op::ShrSI => "ShrSIOp",
            Op::ShrUI => "ShrUIOp",
            Op::SIToFP => "SIToFPOp",
            Op::SubF => "SubFOp",
            Op::SubI => "SubIOp",
            Op::TruncF => "TruncFOp",
            Op::TruncI => "TruncIOp",
            Op::UIToFP => "UIToFPOp",
            Op::XOrI => "XOrIOp",
        })
    }
}

impl fmt::Display for RoundingModeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            RoundingModeKind::ToNearestEven => "to_nearest_even",
            RoundingModeKind::Downward => "downward",
            RoundingModeKind::Upward => "upward",
            RoundingModeKind::TowardZero => "toward_zero",
            RoundingModeKind::ToNearestAway => "to_nearest_away",
        })
    }
}
