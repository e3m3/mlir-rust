// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirOperation;

use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::bool::Bool as BoolAttr;
use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::CustomAttributeData;
use attributes::specialized::NamedAffineMap;
use attributes::specialized::NamedArrayOfBools;
use attributes::specialized::NamedArrayOfIntegers;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedParsed;
use attributes::specialized::NamedString;
use dialects::common::NonTemporal;
use dialects::common::OperandSegmentSizes;
use dialects::IROp;
use dialects::IROperation;
use effects::MemoryEffectList;
use effects::MEFF_DEFAULT_WRITE;
use effects::MEFF_NO_MEMORY_EFFECT;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::OperationState;
use ir::StringBacked;
use ir::Shape;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::IRType;
use types::integer::Integer as IntegerType;
use types::ranked_tensor::RankedTensor;
use types::shaped::Shaped;
use types::vector::Vector;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct InBounds(MlirAttribute);

#[derive(Clone)]
pub struct Offsets(MlirAttribute);

#[derive(Clone)]
pub struct PermutationMap(MlirAttribute);

#[derive(Clone)]
pub struct Punctuation(MlirAttribute);

#[derive(Clone)]
pub struct Sizes(MlirAttribute);

#[derive(Clone)]
pub struct StaticPosition(MlirAttribute);

#[derive(Clone)]
pub struct Strides(MlirAttribute);

#[derive(Clone)]
pub struct StringLiteral(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum Op {
    Bitcast,
    Broadcast,
    CompressStore,
    ConstantMask,
    Contract,
    CreateMask,
    Deinterleave,
    ExpandLoad,
    Extract,
    ExtractElement,
    ExtractStridedSlice,
    Fma,
    FlatTranspose,
    FromElements,
    Gather,
    Insert,
    InsertElement,
    InsertStridedSlice,
    Interleave,
    Load,
    Mask,
    MaskedLoad,
    MaskedStore,
    MatrixMultiply,
    MultiReduction,
    OuterProduct,
    Print,
    Reduction,
    ScalableExtract,
    ScalableInsert,
    Scan,
    Scatter,
    ShapeCast,
    Shuffle,
    Splat,
    Step,
    Store,
    TransferRead,
    TransferWrite,
    Transpose,
    TypeCast,
    VScale,
    WarpExecuteOnLane0,
    Yield,
}

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum PunctuationKind {
    NoPunctuation   = 0,
    NewLine         = 1,
    Comma           = 2,
    Open            = 3,
    Close           = 4,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Extract(MlirOperation);

#[derive(Clone)]
pub struct ExtractElement(MlirOperation);

#[derive(Clone)]
pub struct FromElements(MlirOperation);

#[derive(Clone)]
pub struct Load(MlirOperation);

#[derive(Clone)]
pub struct Print(MlirOperation);

#[derive(Clone)]
pub struct Store(MlirOperation);

#[derive(Clone)]
pub struct TransferRead(MlirOperation);

#[derive(Clone)]
pub struct TransferWrite(MlirOperation);

#[derive(Clone)]
pub struct VectorMask(MlirOperation);

///////////////////////////////
//  Support
///////////////////////////////

#[derive(Clone,Copy,PartialEq)]
pub struct VectorMaskShape(usize);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl InBounds {
    pub fn new(context: &Context, elements: &[bool]) -> Self {
        let attrs: Vec<BoolAttr> = elements.iter().map(|&b| BoolAttr::new(context, b)).collect();
        <Self as NamedArrayOfBools>::new(context, &attrs)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn has_out_of_bounds(&self) -> bool {
        self.as_bools().iter().any(|b| b.get_value())
    }
}

impl Offsets {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Punctuation {
    pub fn new(context: &Context, k: PunctuationKind) -> Self {
        let cad = CustomAttributeData::new(
            Self::get_name().to_string(),
            context.get_dialect_vector().get_namespace().to_string(),
            vec![k.get_name().to_string()],
        );
        <Self as NamedParsed>::new_custom(context, &cad)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_kind(&self) -> PunctuationKind {
        eprintln!("get_kind for PunctuationKind unimplemented");
        exit(ExitCode::DialectError);
    }
}

impl PermutationMap {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Sizes {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticPosition {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn num_static(&self) -> isize {
        self.as_dense_array().num_elements() - self.num_symbolic()
    }

    pub fn num_symbolic(&self) -> isize {
        let a = self.as_dense_array();
        (0..a.num_elements()).filter(|&i| a.get_element_i64(i) == Self::symbolic_pos()).count() as isize
    }

    #[inline]
    pub const fn symbolic_pos() -> i64 {
        i64::MIN
    }
}

impl Strides {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StringLiteral {
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
            Op::Bitcast             => "bitcast",
            Op::Broadcast           => "broadcast",
            Op::CompressStore       => "compressstore",
            Op::ConstantMask        => "constant_mask",
            Op::Contract            => "contract",
            Op::CreateMask          => "create_mask",
            Op::Deinterleave        => "deinterleave",
            Op::ExpandLoad          => "expandload",
            Op::Extract             => "extract",
            Op::ExtractElement      => "extractelement",
            Op::ExtractStridedSlice => "extract_strided_slice",
            Op::Fma                 => "fma",
            Op::FlatTranspose       => "flat_transpose",
            Op::FromElements        => "from_elements",
            Op::Gather              => "gather",
            Op::Insert              => "insert",
            Op::InsertElement       => "insertelement",
            Op::InsertStridedSlice  => "insert_strided_slice",
            Op::Interleave          => "interleave",
            Op::Load                => "load",
            Op::Mask                => "mask",
            Op::MaskedLoad          => "maskedload",
            Op::MaskedStore         => "maskedstore",
            Op::MatrixMultiply      => "matrix_multiply",
            Op::MultiReduction      => "multi_reduction",
            Op::OuterProduct        => "outerproduct",
            Op::Print               => "print",
            Op::Reduction           => "reduction",
            Op::ScalableExtract     => "scalable.extract",
            Op::ScalableInsert      => "scalable.insert",
            Op::Scan                => "scan",
            Op::Scatter             => "scatter",
            Op::ShapeCast           => "shape_cast",
            Op::Shuffle             => "shuffle",
            Op::Splat               => "splat",
            Op::Step                => "step",
            Op::Store               => "store",
            Op::TransferRead        => "transfer_read",
            Op::TransferWrite       => "transfer_write",
            Op::Transpose           => "transpose",
            Op::TypeCast            => "type_cast",
            Op::VScale              => "vscale",
            Op::WarpExecuteOnLane0  => "warp_execute_on_lane_0",
            Op::Yield               => "yield",
        }
    }
}

impl PunctuationKind {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0   => PunctuationKind::NoPunctuation,
            1   => PunctuationKind::NewLine,
            2   => PunctuationKind::Comma,
            3   => PunctuationKind::Open,
            4   => PunctuationKind::Close,
            _   => {
                eprintln!("Invalid value '{}' for punctuation kind", n);
                exit(ExitCode::DialectError);
            },
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            PunctuationKind::NoPunctuation  => "no_punctuation",
            PunctuationKind::NewLine        => "newline",
            PunctuationKind::Comma          => "comma",
            PunctuationKind::Open           => "open",
            PunctuationKind::Close          => "close",
        }
    }
}

///////////////////////////////
//  Support Implementation
///////////////////////////////

impl VectorMaskShape {
    pub fn new(n: usize) -> Self {
        Self::from(n)
    }

    pub fn get(&self) -> usize {
        self.0
    }

    pub fn get_mut(&mut self) -> &mut usize {
        &mut self.0
    }
}

///////////////////////////////
//  Operation Implemention
///////////////////////////////

impl Extract {
    pub fn new(
        t: &Type,
        source: &Value,
        pos: &[Value],
        static_pos: &StaticPosition,
        loc: &Location,
    ) -> Self {
        let t_source = source.get_type();
        if !t_source.is_vector() {
            eprintln!("Expected vector type for source operand of extract operation");
            exit(ExitCode::DialectError);
        }
        if pos.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for dynamic position operand(s) of extract operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*t_source.get());
        let s_source_rank = s_source.rank().unwrap_or(-1);
        let n_static = static_pos.num_elements() as i64;
        let n_result = s_source_rank - n_static;
        if !t.is_vector() && n_result != 0 {
            eprintln!("Expected element type result for source vector operand with rank ({}) \
                equal to the arity of the static position attribute ({}) of extract operation",
                s_source_rank,
                n_static,
            );
            exit(ExitCode::DialectError);
        } else if !t.is_vector() && *t != s_source.get_element_type() {
            eprintln!("Expected matching element type for source operand and result type \
                of extract operation"
            );
            exit(ExitCode::DialectError);
        } else if t.is_vector() {
            let s = Shaped::from(*t.get());
            if s.get_element_type() != s_source.get_element_type() {
                eprintln!("Expected matching element type for source operand and result type \
                    of extract operation"
                );
                exit(ExitCode::DialectError);
            }
            let s_rank = s.rank().unwrap_or(-1);
            if s_rank != n_result {
                eprintln!("Expected rank of vector type result ({}) to be equal to the difference ({}) \
                    of the rank of the source vector type ({}) and the arity of the \
                    static position attribute ({}) for extract operation",
                    s_rank,
                    n_result,
                    s_source_rank,
                    n_static,
                );
                exit(ExitCode::DialectError);
            }
        } else {
            eprintln!("Expected vector type or element type of source operand for result type \
                of extract operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Extract.get_name(),
        ));
        let mut args = vec![source.clone()];
        args.append(&mut pos.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[static_pos.as_named_attribute()]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Extract(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl ExtractElement {
    pub fn new(t: &Type, source: &Value, pos: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        let t_pos = pos.get_type();
        if !t_source.is_vector() {
            eprintln!("Expected vector type for source operand of extract element operation");
            exit(ExitCode::DialectError);
        }
        if !t_pos.is_index() && (!t_pos.is_integer() || !IntegerType::from(*t_pos.get()).is_signless()) {
            eprintln!("Expected index or signless integer type for position operand \
                of extract element operation"
            );
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*t_source.get());
        if *t != s_source.get_element_type() {
            eprintln!("Expected matching element type for source operand and result type \
                of extract element operation"
            );
            exit(ExitCode::DialectError);
        }
        if s_source.rank().unwrap_or(-1) != 1 {
            eprintln!("Expected matching 1-D vector for source operand of extract element operation");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::ExtractElement.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone(), pos.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_0_d(t: &Type, source: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        if !t_source.is_vector() {
            eprintln!("Expected vector type for source operand of extract element operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*t_source.get());
        if *t != s_source.get_element_type() {
            eprintln!("Expected matching element type for source operand and result type \
                of extract element operation"
            );
            exit(ExitCode::DialectError);
        }
        if s_source.rank().unwrap_or(-1) != 0 {
            eprintln!("Expected matching 0-D vector for source operand of extract element operation");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::ExtractElement.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        ExtractElement(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl FromElements {
    pub fn new(t: &Vector, args: &[Value], loc: &Location) -> Self {
        let s = t.as_shaped();
        let n = s.num_elements().unwrap_or(0);
        if n <= 0 || args.is_empty() {
            eprintln!("Expected non-empty result types and arguments for from elements");
            exit(ExitCode::DialectError);
        }
        if n != args.len() as i64 {
            eprintln!("Expected matching number of result types and arguments for from elements");
            exit(ExitCode::DialectError);
        }
        let t_ = s.get_element_type();
        if args.iter().any(|a| t_ != a.get_type()) {
            eprintln!("Expected matching number of result and argument types for from elements");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::FromElements.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        FromElements(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Load {
    pub fn new(
        t: &Vector,
        base: &Value,
        indices: &[Value],
        is_nt: &NonTemporal,
        loc: &Location,
    ) -> Self {
        if !base.get_type().is_mem_ref() {
            eprintln!("Expected memory reference type for base operand of load operation");
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices operand(s) of load operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_base = Shaped::from(*base.get_type().get());
        let t_base_elem = s_base.get_element_type();
        if s.get_element_type() != t_base_elem && t.as_type() != t_base_elem {
            eprintln!("Expected matching types for element type of base operand and result type \
                of load operation"
            );
            exit(ExitCode::DialectError);
        }
        let s_rank_base = s_base.rank().unwrap_or(-1);
        let n_indices = indices.len() as i64 ;
        if s_rank_base != n_indices {
            eprintln!("Expected number of indices ({}) to match rank of base operand ({}) \
                of load operation",
                n_indices,
                s_rank_base,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Load.get_name(),
        ));
        let mut args = vec![base.clone()];
        args.append(&mut indices.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[is_nt.as_named_attribute()]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Load(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Print {
    pub fn new(context: &Context, args: &[Value], p: PunctuationKind, loc: &Location) -> Self {
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Print.get_name(),
        ));
        let punc_attr = Punctuation::new(context, p);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[punc_attr.as_named_attribute()]);
        op_state.add_operands(args);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_string(context: &Context, s: &StringLiteral, loc: &Location) -> Self {
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Print.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[s.as_named_attribute()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Print(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_punctuation(&self) -> Option<Punctuation> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(Punctuation::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr_name = StringBacked::from(Punctuation::get_name());
            let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
            Some(Punctuation::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_punctuation_kind(&self) -> Option<PunctuationKind> {
        self.get_punctuation().map(|p| p.get_kind())
    }

    pub fn get_string_literal(&self) -> Option<StringLiteral> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(StringLiteral::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr_name = StringBacked::from(StringLiteral::get_name());
            let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
            Some(StringLiteral::from(*attr.get()))
        } else {
            None
        }
    }
}

impl Store {
    pub fn new(
        context: &Context,
        value: &Value,
        base: &Value,
        indices: &[Value],
        is_nt: &NonTemporal,
        loc: &Location,
    ) -> Self {
        if !value.get_type().is_vector() {
            eprintln!("Expected vector type for value operand of store operation");
            exit(ExitCode::DialectError);
        }
        if !base.get_type().is_mem_ref() {
            eprintln!("Expected memory reference type for base operand of store operation");
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices operand(s) of store operation");
            exit(ExitCode::DialectError);
        }
        let t = value.get_type();
        let s = Shaped::from(*t.get());
        let s_base = Shaped::from(*base.get_type().get());
        let t_base_elem = s_base.get_element_type();
        if s.get_element_type() != t_base_elem && t != t_base_elem {
            eprintln!("Expected matching element type of base operand and value operand type \
                of store operation"
            );
            exit(ExitCode::DialectError);
        }
        let rank_value = s.rank().unwrap_or(-1);
        let rank_base = s_base.rank().unwrap_or(-1);
        if rank_base < rank_value {
            eprintln!("Expected rank of base operand ({}) to be greater than or equal to rank \
                of value operand ({}) of store operation",
                rank_base,
                rank_value,
            );
            exit(ExitCode::DialectError);
        }
        let n_indices = indices.len() as i64;
        if rank_base != n_indices {
            eprintln!("Expected number of indices ({}) to match rank of base operand ({}) \
                of store operation",
                n_indices,
                rank_base,
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Store.get_name(),
        ));
        let mut args = vec![value.clone(), base.clone()];
        args.append(&mut indices.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[is_nt.as_named_attribute()]);
        op_state.add_operands(&args);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Store(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl TransferRead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: &Vector,
        bounds_attr: &InBounds,
        perm_attr: &PermutationMap,
        source: &Value,
        indices: &[Value],
        padding: &Value,
        mask: Option<&Value>,
        loc: &Location,
    ) -> Self {
        if !source.get_type().is_shaped() {
            eprintln!("Expected shaped source operand for transfer read operation");
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected indices for transfer read operation");
            exit(ExitCode::DialectError);
        }
        if let Some(mask_) = mask {
            if !VectorMask::is_mask_type(&mask_.get_type()) {
                eprintln!("Expected vector mask type for transfer read operation");
                exit(ExitCode::DialectError);
            }
        }
        let s = t.as_shaped();
        let s_source = Shaped::from(*source.get_type().get());
        let t_source_elem = s_source.get_element_type();
        // Check for matching source and result element type or if source element type is a suffix
        // of the result type.
        if t_source_elem.is_vector() {
            let s_source_elem = Vector::from(*t_source_elem.get()).as_shaped();
            if let Some(s_suffix) = s.get_matching_suffix(&s_source_elem) {
                if s_suffix.unpack() != s_source_elem.unpack() {
                    eprintln!("Expected matching source element type and result type suffix \
                        for transfer read operation"
                    );
                    exit(ExitCode::DialectError);
                }
            } else {
                eprintln!("Expected matching source element type and result type suffix \
                    for transfer read operation"
                );
                exit(ExitCode::DialectError);
            }
        } else if t.as_type() != t_source_elem && s.get_element_type() != t_source_elem {
            eprintln!("Expected matching source element type and result element type \
                for transfer read operation"
            );
            exit(ExitCode::DialectError);
        }
        if padding.get_type() != s_source.get_element_type() {
            eprintln!("Expected matching source element type and padding type for \
                transfer read operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::TransferRead.get_name(),
        ));
        let mut args: Vec<Value> = Vec::new();
        let mut opseg_sizes: Vec<i32> = Vec::new();
        args.push(source.clone());
        opseg_sizes.push(1);
        args.append(&mut indices.to_vec());
        opseg_sizes.push(indices.len() as i32);
        args.push(padding.clone());
        opseg_sizes.push(1);
        if let Some(mask_) = mask.cloned() {
            args.push(mask_);
            opseg_sizes.push(1);
        } else {
            opseg_sizes.push(0);
        }
        let opseg_attr = OperandSegmentSizes::new(&context, &opseg_sizes);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            opseg_attr.as_named_attribute(),
            bounds_attr.as_named_attribute(),
            perm_attr.as_named_attribute(),
        ]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        TransferRead(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_bounds_attribute(&self) -> InBounds {
        let attr_name = StringBacked::from(InBounds::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        InBounds::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_permuration_attribute(&self) -> PermutationMap {
        let attr_name = StringBacked::from(PermutationMap::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        PermutationMap::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl TransferWrite {
    #[allow(clippy::too_many_arguments)]
    pub fn new_memref(
        context: &Context,
        bounds_attr: &InBounds,
        perm_attr: &PermutationMap,
        value: &Value,
        source: &Value,
        indices: &[Value],
        mask: Option<&Value>,
        loc: &Location,
    ) -> Self {
        if let Err(msg) = Self::check_types(None, value, source, indices, mask) {
            eprintln!("{}", msg);
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::TransferWrite.get_name(),
        ));
        let mut args: Vec<Value> = Vec::new();
        let mut opseg_sizes: Vec<i32> = Vec::new();
        args.push(value.clone());
        opseg_sizes.push(1);
        args.push(source.clone());
        opseg_sizes.push(1);
        args.append(&mut indices.to_vec());
        opseg_sizes.push(indices.len() as i32);
        if let Some(mask_) = mask.cloned() {
            args.push(mask_);
            opseg_sizes.push(1);
        } else {
            opseg_sizes.push(0);
        }
        let opseg_attr = OperandSegmentSizes::new(context, &opseg_sizes);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            opseg_attr.as_named_attribute(),
            bounds_attr.as_named_attribute(),
            perm_attr.as_named_attribute(),
        ]);
        op_state.add_operands(&args);
        Self::from(*op_state.create_operation().get())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_tensor(
        t: &RankedTensor,
        bounds_attr: &InBounds,
        perm_attr: &PermutationMap,
        value: &Value,
        source: &Value,
        indices: &[Value],
        mask: Option<&Value>,
        loc: &Location,
    ) -> Self {
        if let Err(msg) = Self::check_types(Some(t), value, source, indices, mask) {
            eprintln!("{}", msg);
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_vector();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::TransferWrite.get_name(),
        ));
        let mut args: Vec<Value> = Vec::new();
        let mut opseg_sizes: Vec<i32> = Vec::new();
        args.push(value.clone());
        opseg_sizes.push(1);
        args.push(source.clone());
        opseg_sizes.push(1);
        args.append(&mut indices.to_vec());
        opseg_sizes.push(indices.len() as i32);
        if let Some(mask_) = mask.cloned() {
            args.push(mask_);
            opseg_sizes.push(1);
        } else {
            opseg_sizes.push(0);
        }
        let opseg_attr = OperandSegmentSizes::new(&context, &opseg_sizes);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            opseg_attr.as_named_attribute(),
            bounds_attr.as_named_attribute(),
            perm_attr.as_named_attribute(),
        ]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        TransferWrite(op)
    }

    fn check_types(
        result: Option<&RankedTensor>,
        value: &Value,
        source: &Value,
        indices: &[Value],
        mask: Option<&Value>,
    ) -> Result<(), String> {
        if !value.get_type().is_vector() {
            Err("Expected vector for value operand of transfer write operation".to_string())?
        }
        if !source.get_type().is_shaped() {
            Err("Expected shaped type for source operand of transfer write operation".to_string())?
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            Err("Expected indices for transfer write operation".to_string())?
        }
        if let Some(mask_) = mask {
            if !VectorMask::is_mask_type(&mask_.get_type()) {
                Err("Expected vector mask type for transfer write operation".to_string())?
            }
        }
        let t_value = value.get_type();
        let t_source = source.get_type();
        let s_value = Shaped::from(*t_value.get());
        let s_source = Shaped::from(*t_source.get());
        let t_source_elem = s_source.get_element_type();
        if let Some(t_) = result {
            if t_.as_type() != t_source {
                return Err("Expected matching source operand type and result type for \
                    transfer write operation".to_string()
                );
            }
        }
        if t_source_elem.is_vector() {
            let s_source_elem = Vector::from(*t_source_elem.get()).as_shaped();
            if let Some(s_suffix) = s_value.get_matching_suffix(&s_source_elem) {
                return if s_suffix.unpack() != s_source_elem.unpack() {
                    Err("Expected matching source element type and result type suffix \
                        for transfer write operation".to_string()
                    )
                } else {
                    Ok(())
                };
            } else {
                return Err("Expected matching source element type and result type suffix \
                    for transfer write operation".to_string()
                );
            }
        }
        if s_value.get_element_type() != s_source.get_element_type() {
            return Err("Expected matching element types for source operand and value operand \
                of transfer write operation".to_string()
            );
        }
        let rank_value = s_value.rank().unwrap_or(-1);
        let rank_source = s_source.rank().unwrap_or(-1);
        if rank_source == 0 && rank_value == 1 {
            // Rank 1 vector matches Rank 0 Tensor (i.e., write vector<1xt> into tensor<t>)
            Ok(())
        } else if rank_value > rank_source {
            Err(format!(
                "Expected rank for source operand type ({}) to be greater than or equal to \
                the rank of the value operand type ({}) of transfer write operation",
                rank_source,
                rank_value,
            ))
        } else {
            Ok(())
        }
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_bounds_attribute(&self) -> InBounds {
        let attr_name = StringBacked::from(InBounds::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        InBounds::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_permutation_attribute(&self) -> PermutationMap {
        let attr_name = StringBacked::from(PermutationMap::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        PermutationMap::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl VectorMask {
    const WIDTH: usize = 1;

    pub fn new(context: &Context, values: &[Value], loc: &Location) -> Self {
        if values.is_empty() {
            eprintln!("Expected non-empty values for vector mask");
            exit(ExitCode::DialectError);
        }
        if values.iter().any(|v| !Self::is_mask_elem(&v.get_type())) {
            eprintln!("Expected 1-bit signless integer values for vector mask");
            exit(ExitCode::DialectError);
        }
        let t = Self::get_type(context, values.len());
        Self::from(*FromElements::new(&t, values, loc).get_mut())
    }

    pub fn from(op: MlirOperation) -> Self {
        VectorMask(op)
    }

    pub fn as_from_elements(&self) -> FromElements {
        FromElements::from(*self.get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_type(context: &Context, size: usize) -> Vector {
        let shape = VectorMaskShape::new(size);
        let t = IntegerType::new_signless(context, Self::WIDTH);
        Vector::new(&shape, &t.as_type())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn is_mask_elem(t: &Type) -> bool {
        if !t.is_integer() {
            return false;
        }
        let t_int = IntegerType::from(*t.get());
        t_int.is_signless() && t_int.get_width() == Self::WIDTH
    }

    pub fn is_mask_type(t: &Type) -> bool {
        t.is_vector() && Self::is_mask_vector(&Vector::from(*t.get()))
    }

    pub fn is_mask_vector(t: &Vector) -> bool {
        let s = t.as_shaped();
        let n = s.num_elements().unwrap_or(0);
        n > 0 && Self::is_mask_elem(&s.get_element_type())
    }
}

///////////////////////////////
//  Trait Implemention
///////////////////////////////

impl IROperation for Extract {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
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
        Op::Extract.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Extract
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::InferTypeOpAdaptor,
        ]
    }
}

impl IROperation for ExtractElement {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
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
        Op::ExtractElement.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::ExtractElement
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for FromElements {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::FromElements.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::FromElements
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for Load {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Load.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Load
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirAttribute> for InBounds {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for InBounds {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for InBounds {
    fn get_name() -> &'static str {
        "in_bounds"
    }
}

impl NamedArrayOfBools for InBounds {}

impl From<MlirAttribute> for Offsets {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for Offsets {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Offsets {
    fn get_name() -> &'static str {
        "offsets"
    }
}

impl NamedArrayOfIntegers for Offsets {}

impl IROp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl IROperation for Print {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_DEFAULT_WRITE,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::MemoryEffect(MemoryEffectOpInterface::MemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Print.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Print
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirAttribute> for PermutationMap {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for PermutationMap {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for PermutationMap {
    fn get_name() -> &'static str {
        "permutation_map"
    }
}

impl NamedAffineMap for PermutationMap {}

impl From<MlirAttribute> for Punctuation {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for Punctuation {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Punctuation {
    fn get_name() -> &'static str {
        "punctuation"
    }
}

impl NamedParsed for Punctuation {}

impl From<i32> for PunctuationKind {
    fn from(n: i32) -> Self {
        PunctuationKind::from_i32(n)
    }
}

impl From<String> for PunctuationKind {
    fn from(s: String) -> Self {
        PunctuationKind::from(&s)
    }
}

impl From<&String> for PunctuationKind {
    fn from(s: &String) -> Self {
        match PunctuationKind::from_str(s.as_str()) {
            Ok(k)       => k,
            Err(msg)    => {
                eprintln!("{}", msg);
                exit(ExitCode::DialectError);
            },
        }
    }
}

impl FromStr for PunctuationKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "no_punctuation"    => Ok(PunctuationKind::NoPunctuation),
            "newline"           => Ok(PunctuationKind::NewLine),
            "comma"             => Ok(PunctuationKind::Comma),
            "open"              => Ok(PunctuationKind::Open),
            "close"             => Ok(PunctuationKind::Close),
            _   => {
                Err(format!("Invalid value '{}' for punctuation kind", s))
            },
        }
    }
}

impl From<MlirAttribute> for Sizes {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for Sizes {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Sizes {
    fn get_name() -> &'static str {
        "sizes"
    }
}

impl NamedArrayOfIntegers for Sizes {}

impl IROperation for Store {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Store.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Store
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirAttribute> for Strides {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for Strides {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Strides {
    fn get_name() -> &'static str {
        "strides"
    }
}

impl NamedArrayOfIntegers for Strides {}

impl From<MlirAttribute> for StaticPosition {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for StaticPosition {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticPosition {
    fn get_name() -> &'static str {
        "static_position"
    }
}

impl NamedI64DenseArray for StaticPosition {}

impl From<MlirAttribute> for StringLiteral {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for StringLiteral {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StringLiteral {
    fn get_name() -> &'static str {
        "stringLiteral"
    }
}

impl NamedString for StringLiteral {}

impl IROperation for TransferRead {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::MaskableOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::UndefinedMemoryEffect),
            Interface::VectorTransferOpInterface,
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::TransferRead.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::TransferRead
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
        ]
    }
}

impl IROperation for TransferWrite {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_vector()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::MaskableOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::UndefinedMemoryEffect),
            Interface::VectorTransferOpInterface,
            Interface::VectorUnrollOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::TransferWrite.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::TransferWrite
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
        ]
    }
}

impl IROperation for VectorMask {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_from_elements().get_dialect()
    }

    fn get_effects(&self) -> MemoryEffectList {
        self.as_from_elements().get_effects()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        self.as_from_elements().get_interfaces()
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        self.as_from_elements().get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        self.as_from_elements().get_op()
    }

    fn get_traits(&self) -> &'static [Trait] {
        self.as_from_elements().get_traits()
    }
}

impl From<usize> for VectorMaskShape {
    fn from(n: usize) -> Self {
        if n == 0 {
            eprintln!("Expected no empty vector mask shape");
            exit(ExitCode::DialectError);
        }
        VectorMaskShape(n)
    }
}

impl Shape for VectorMaskShape {
    fn rank(&self) -> isize {
        1
    }

    fn get(&self, i: isize) -> i64 {
        match i {
            0   => self.get() as i64,
            _   => {
                eprintln!("Index '{}' out of bounds for vector mask shape", i);
                exit(ExitCode::DialectError);
            },
        }
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Bitcast             => "BitcastOp",
            Op::Broadcast           => "BroadcastOp",
            Op::CompressStore       => "CompressStoreOp",
            Op::ConstantMask        => "ConstantMaskOp",
            Op::Contract            => "ContractOp",
            Op::CreateMask          => "CreateMaskOp",
            Op::Deinterleave        => "DeinterleaveOp",
            Op::ExpandLoad          => "ExpandLoadOp",
            Op::Extract             => "ExtractOp",
            Op::ExtractElement      => "ExtractElementOp",
            Op::ExtractStridedSlice => "ExtractStridedSliceOp",
            Op::Fma                 => "FmaOp",
            Op::FlatTranspose       => "FlatTransposeOp",
            Op::FromElements        => "FromElementsOp",
            Op::Gather              => "GatherOp",
            Op::Insert              => "InsertOp",
            Op::InsertElement       => "InsertElementOp",
            Op::InsertStridedSlice  => "InsertStridedSliceOp",
            Op::Interleave          => "InterleaveOp",
            Op::Load                => "LoadOp",
            Op::Mask                => "MaskOp",
            Op::MaskedLoad          => "MaskedLoadOp",
            Op::MaskedStore         => "MaskedStoreOp",
            Op::MatrixMultiply      => "MatrixMultiplyOp",
            Op::MultiReduction      => "MultiReductionOp",
            Op::OuterProduct        => "OuterProductOp",
            Op::Print               => "PrintOp",
            Op::Reduction           => "ReductionOp",
            Op::ScalableExtract     => "ScalableExtractOp",
            Op::ScalableInsert      => "ScalableInsertOp",
            Op::Scan                => "ScanOp",
            Op::Scatter             => "ScatterOp",
            Op::ShapeCast           => "ShapeCastOp",
            Op::Shuffle             => "ShuffleOp",
            Op::Splat               => "SplatOp",
            Op::Step                => "StepOp",
            Op::Store               => "StoreOp",
            Op::TransferRead        => "TransferReadOp",
            Op::TransferWrite       => "TransferWriteOp",
            Op::Transpose           => "TransposeOp",
            Op::TypeCast            => "TypeCastOp",
            Op::VScale              => "VScaleOp",
            Op::WarpExecuteOnLane0  => "WarpExecuteOnLane0Op",
            Op::Yield               => "YieldOp",
        })
    }
}

impl fmt::Display for PunctuationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            PunctuationKind::NoPunctuation  => "NoPunctuation",
            PunctuationKind::NewLine        => "NewLine",
            PunctuationKind::Comma          => "Comma",
            PunctuationKind::Open           => "Open",
            PunctuationKind::Close          => "Close",
        })
    }
}
