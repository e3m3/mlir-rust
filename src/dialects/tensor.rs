// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;
use mlir::MlirOperation;

use std::cmp;
use std::fmt;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::array::Array;
use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::NamedArrayOfIntegerArrays;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedUnit;
use dialects::common::Dimension;
use dialects::common::OperandSegmentSizes;
use dialects::common::StaticOffsets;
use dialects::common::StaticSizes;
use dialects::common::StaticStrides;
use dialects::IROp;
use dialects::IROperation;
use effects::MemoryEffectList;
use effects::MEFF_NO_MEMORY_EFFECT;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Block;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::Operation;
use ir::OperationState;
use ir::Region;
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::index::Index;
use types::IRType;
use types::ranked_tensor::RankedTensor;
use types::shaped::Shaped;
use types::unranked_tensor::UnrankedTensor;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct GatherDimensions(MlirAttribute);

#[derive(Clone)]
pub struct InnerDimensionsPosition(MlirAttribute);

#[derive(Clone)]
pub struct NoFold(MlirAttribute);

#[derive(Clone)]
pub struct OuterDimensionsPermutation(MlirAttribute);

#[derive(Clone)]
pub struct Reassociation(MlirAttribute);

#[derive(Clone)]
pub struct ScatterDimensions(MlirAttribute);

#[derive(Clone)]
pub struct StaticHigh(MlirAttribute);

#[derive(Clone)]
pub struct StaticInnerTiles(MlirAttribute);

#[derive(Clone)]
pub struct StaticLow(MlirAttribute);

#[derive(Clone)]
pub struct StaticOutputShape(MlirAttribute);

#[derive(Clone)]
pub struct Unique(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum Op {
    Bitcast,
    Cast,
    CollapseShape,
    Concat,
    Dim,
    Empty,
    ExpandShape,
    Extract,
    ExtractSlice,
    FromElements,
    Gather,
    Generate,
    Insert,
    InsertSlice,
    Pack,
    Pad,
    ParallelInsertSlice,
    Rank,
    Reshape,
    Scatter,
    Splat,
    Unpack,
    Yield,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Bitcast(MlirOperation);

#[derive(Clone)]
pub struct Cast(MlirOperation);

#[derive(Clone)]
pub struct CollapseShape(MlirOperation);

#[derive(Clone)]
pub struct Concat(MlirOperation);

#[derive(Clone)]
pub struct Dim(MlirOperation);

#[derive(Clone)]
pub struct Empty(MlirOperation);

#[derive(Clone)]
pub struct ExpandShape(MlirOperation);

#[derive(Clone)]
pub struct Extract(MlirOperation);

#[derive(Clone)]
pub struct ExtractSlice(MlirOperation);

#[derive(Clone)]
pub struct FromElements(MlirOperation);

#[derive(Clone)]
pub struct Generate(MlirOperation);

#[derive(Clone)]
pub struct Pad(MlirOperation);

#[derive(Clone)]
pub struct Rank(MlirOperation);

#[derive(Clone)]
pub struct Reshape(MlirOperation);

#[derive(Clone)]
pub struct Yield(MlirOperation, MlirOperation, Op);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl GatherDimensions {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl InnerDimensionsPosition {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl NoFold {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl OuterDimensionsPermutation {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Reassociation {
    pub fn new(context: &Context, values: &[Array]) -> Self {
        const WIDTH: usize = 64;
        <Self as NamedArrayOfIntegerArrays>::new(context, values, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl ScatterDimensions {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticHigh {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticInnerTiles {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticOutputShape {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn num_dynamic_dims(&self) -> isize {
        let a = self.as_dense_array();
        (0..a.num_elements())
            .filter(|&i| a.get_element_i64(i) == Shaped::dynamic_size())
            .count()
            as isize
    }

    pub fn num_static_dims(&self) -> isize {
        self.as_dense_array().num_elements() - self.num_dynamic_dims()
    }
}

impl StaticLow {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Unique {
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
            Op::Cast                => "cast",
            Op::CollapseShape       => "collapse_shape",
            Op::Concat              => "concat",
            Op::Dim                 => "dim",
            Op::Empty               => "empty",
            Op::ExpandShape         => "expand_shape",
            Op::Extract             => "extract",
            Op::ExtractSlice        => "extract_slice",
            Op::FromElements        => "from_elements",
            Op::Gather              => "gather",
            Op::Generate            => "generate",
            Op::Insert              => "insert",
            Op::InsertSlice         => "insert_slice",
            Op::Pack                => "pack",
            Op::Pad                 => "pad",
            Op::ParallelInsertSlice => "parallel_insert_slice",
            Op::Rank                => "rank",
            Op::Reshape             => "reshape",
            Op::Scatter             => "scatter",
            Op::Splat               => "splat",
            Op::Unpack              => "unpack",
            Op::Yield               => "yield",
        }
    }
}

///////////////////////////////
//  Operation Implemention
///////////////////////////////

impl Bitcast {
    fn new(t: &Shaped, t_value: &Shaped, value: &Value, loc: &Location) -> Self {
        if !t.has_matching_element_type_width(t_value) {
            eprintln!("Expected matching element type widths for source and result tensor types");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Bitcast.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_ranked(t: &RankedTensor, value: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor type for bitcast result and operand");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_value = Shaped::from(*t_value.get());
        if t_value.is_ranked_tensor() {
            let t_value_tensor = RankedTensor::from_type(&t_value);
            if !t.has_matching_ranks(&t_value_tensor) {
                eprintln!("Expected matching ranks for ranked tensor type of bitcast operation");
                exit(ExitCode::DialectError);
            }
            if !t.has_matching_static_dimensions(&t_value_tensor) {
                eprintln!("Expected matching sizes for static dimension of bitcast operation");
                exit(ExitCode::DialectError);
            }
        }
        Self::new(&s, &s_value, value, loc)
    }

    pub fn new_unranked(t: &UnrankedTensor, value: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor type for bitcast result and operand");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_value = Shaped::from(*t_value.get());
        Self::new(&s, &s_value, value, loc)
    }

    pub fn from(op: MlirOperation) -> Self {
        Bitcast(op)
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

impl Cast {
    /// TODO:? "The operation is invalid if converting to a mismatching constant dimension." [1]
    /// [1]: https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcast-tensorcastop
    pub fn new(s: &Shaped, s_value: &Shaped, value: &Value, loc: &Location) -> Self {
        if s.get_element_type() != s_value.get_element_type() {
            eprintln!("Expected matching element types for source and result tensor types of \
                cast shape operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = s.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Cast.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[s.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_ranked(t: &RankedTensor, value: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor type for cast result and operand");
            exit(ExitCode::DialectError);
        }
        let s = Shaped::from(*t.get());
        let s_value = Shaped::from(*t_value.get());
        if t_value.is_ranked_tensor() {
            let t_value_tensor = RankedTensor::from_type(&t_value);
            if !t.has_matching_ranks(&t_value_tensor) {
                eprintln!("Expected matching ranks for ranked tensor type cast");
                exit(ExitCode::DialectError);
            }
            if !t.has_matching_static_dimensions(&t_value_tensor) {
                eprintln!("Expected matching sizes for static dimension cast");
                exit(ExitCode::DialectError);
            }
        }
        Self::new(&s, &s_value, value, loc)
    }

    pub fn new_unranked(t: &UnrankedTensor, value: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor type for cast result and operand");
            exit(ExitCode::DialectError);
        }
        let s = Shaped::from(*t.get());
        let s_value = Shaped::from(*t_value.get());
        Self::new(&s, &s_value, value, loc)
    }

    pub fn from(op: MlirOperation) -> Self {
        Cast(op)
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

impl CollapseShape {
    pub fn new(t: &RankedTensor, value: &Value, reassoc: &Reassociation, loc: &Location) -> Self {
        let t_value = value.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor type for result and source operand \
                of collapse shape operation"
            );
            exit(ExitCode::DialectError);
        }
        let s = RankedTensor::from(*t.get()).as_shaped();
        let s_value = RankedTensor::from(*value.get_type().get()).as_shaped();
        if s.get_element_type() != s_value.get_element_type() {
            eprintln!("Expected matching element types for source and result tensor types of \
                collapse shape operation"
            );
            exit(ExitCode::DialectError);
        }
        let rank = s.rank().unwrap_or(-1);
        let rank_value = s_value.rank().unwrap_or(-1);
        let rank_reassoc = reassoc.as_array().num_elements();
        if rank != rank_reassoc as i64 {
            eprintln!("Expected rank of resulting tensor to be equal to reassociation rank");
            exit(ExitCode::DialectError);
        } else if rank > rank_value {
            eprintln!("Expected rank of resulting tensor to be of equal or lesser rank");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::CollapseShape.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[reassoc.as_named_attribute()]);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        CollapseShape(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_reassociation(&self) -> Reassociation {
        let attr_name = StringBacked::from(Reassociation::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Reassociation::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Concat {
    pub fn new(t: &RankedTensor, values: &[Value], dim: &Dimension, loc: &Location) -> Self {
        if values.iter().any(|v| !v.get_type().is_ranked_tensor()) {
            eprintln!("Expected ranked tensor types for concat operands");
            exit(ExitCode::DialectError);
        }
        let t_values: Vec<RankedTensor> = values
            .iter()
            .map(|v| RankedTensor::from(*v.get_type().get()))
            .collect();
        if t_values.iter().any(|t_| !t.has_matching_ranks(t_)) {
            eprintln!("Expected values of same rank as tensor result for concat operation");
            exit(ExitCode::DialectError);
        }
        let d = dim.get_value();
        let s = t.as_shaped();
        let t_elem = s.get_element_type();
        let rank = s.rank().unwrap_or(-1);
        if d < 0 || d >= rank {
            eprintln!("Expected concatenated dimension to fall within range [0..{})", rank);
            exit(ExitCode::DialectError);
        }
        let s_values: Vec<Shaped> = t_values.iter().map(|t_| t_.as_shaped()).collect();
        if s_values.iter().any(|s_| t_elem != s_.get_element_type()) {
            eprintln!("Expected matching element types for tensor values and result tensor type \
                of concat operation"
            );
            exit(ExitCode::DialectError);
        }
        for i in 0..rank {
            let do_concat = i == d;
            let size = s.dim_size(i as isize);
            if do_concat && s_values.iter().all(|s_| !s_.is_dynamic_dim(i as isize)) &&
                size != s_values.iter().fold(0, |acc,s_| acc + s_.dim_size(i as isize)) {
                eprintln!(
                    "Expected result dimension to equal sum of operand sizes \
                    along the concatenated dimension"
                );
                exit(ExitCode::DialectError);
            } else if !do_concat && s_values.iter().any(|s_| size != s_.dim_size(i as isize)) {
                eprintln!(
                    "Expected matching dimension size for result and operands along \
                    the non-concatenated dimensions"
                );
                exit(ExitCode::DialectError);
            }
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Concat.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[dim.as_named_attribute()]);
        op_state.add_operands(values);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Concat(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_dimension(&self) -> Dimension {
        let attr_name = StringBacked::from(Dimension::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Dimension::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Dim {
    pub fn new(context: &Context, value: &Value, index: &Value, loc: &Location) -> Self {
        let t = value.get_type();
        if !t.is_tensor() {
            eprintln!("Expected tensor type for first operand");
            exit(ExitCode::DialectError);
        }
        if t.is_ranked_tensor() {
            let t_tensor = RankedTensor::from_type(&t).as_shaped();
            if t_tensor.rank().unwrap_or(-1) <= 0 {
                eprintln!("Expected non-zero rank for ranked tensor type");
                exit(ExitCode::DialectError);
            }
        }
        if !index.get_type().is_index() {
            eprintln!("Expected index type for second operand");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Dim.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone(), index.clone()]);
        op_state.add_results(&[Index::new(context).as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Dim(op)
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

impl Empty {
    pub fn new(t: &RankedTensor, sizes: &[Value], loc: &Location) -> Self {
        if t.as_shaped().num_dynamic_dims().unwrap_or(-1) != sizes.len() as i64 {
            eprintln!("Expected matching arity of dynamic sizes and number of dynamic dimensions for \
                tensor type of empty tensor operation"
            );
            exit(ExitCode::DialectError);
        }
        if !sizes.iter().all(|v| v.get_type().is_index()) {
            eprintln!("Expected index type for dynamic size operands to empty tensor");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Empty.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(sizes);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Empty(op)
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

    pub fn get_size(&self, i: isize) -> Value {
        self.as_operation().get_operand(i)
    }

    pub fn num_sizes(&self) -> isize {
        self.as_operation().num_operands()
    }
}

impl ExpandShape {
    pub fn new(
        t: &RankedTensor,
        source: &Value,
        shape: &[Value],
        reassoc: &Reassociation,
        static_shape: &StaticOutputShape,
        loc: &Location,
    ) -> Self {
        let t_source = source.get_type();
        if !t_source.is_tensor() {
            eprintln!("Expected tensor type for source operand of expand shape operation");
            exit(ExitCode::DialectError);
        }
        if shape.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index types for output shape operand(s) of expand shape operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_source = Shaped::from(*t_source.get());
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!("Expected matching element type for source and result tensor type \
                of expand shape operation"
            );
            exit(ExitCode::DialectError);
        }
        let rank = s.rank().unwrap_or(-1);
        let rank_source = s_source.rank().unwrap_or(-1);
        if rank < rank_source {
            eprintln!("Expected rank of result tensor type ({}) to be of equal or greater rank than \
                the rank of th esource operand ({}) of expand shape operation",
                rank,
                rank_source,
            );
            exit(ExitCode::DialectError);
        }
        let n_static_shape = static_shape.num_dynamic_dims();
        let n_dyn_sizes = shape.len() as isize;
        if n_static_shape != n_dyn_sizes {
            eprintln!("Expected matching number of symbolic dimensions ({}) (length of output shape) \
                to match number of dynamic dimensions (value '{}') in static output shape ({}) \
                of expand shape operation",
                n_dyn_sizes,
                Shaped::dynamic_size(),
                n_static_shape,
            );
            exit(ExitCode::DialectError);
        }
        let n_reassoc = reassoc.num_elements() as i64;
        let n_reassoc_flat = reassoc.num_elements_flattened() as i64;
        if rank != n_reassoc_flat {
            eprintln!("Expected result tensor of rank ({}) equal to number of \
                total dimensions (flattened) specified by the reassociation map ({}) \
                of expand shape operation",
                rank,
                n_reassoc_flat,
            );
            exit(ExitCode::DialectError);
        }
        if rank_source != n_reassoc {
            eprintln!("Expected source tensor of rank equal ({}) to number of dimensional groupings \
                specified by the reassociation map ({}) of expand shape operation",
                rank_source,
                n_reassoc,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::ExpandShape.get_name(),
        ));
        let mut args = vec![source.clone()];
        args.append(&mut shape.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            reassoc.as_named_attribute(),
            static_shape.as_named_attribute(),
        ]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        ExpandShape(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_reassociation(&self) -> Reassociation {
        let attr_name = StringBacked::from(Reassociation::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Reassociation::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_static_output_shape(&self) -> StaticOutputShape {
        let attr_name = StringBacked::from(StaticOutputShape::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticOutputShape::from(*attr.get())
    }
}

impl Extract {
    pub fn new(t: &Type, source: &Value, indices: &[Value], loc: &Location) -> Self {
        if !source.get_type().is_tensor() {
            eprintln!("Expected tensor type for source operand of extract operation");
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of extract operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*source.get_type().get());
        if *t != s_source.get_element_type() {
            eprintln!("Expected matching result type and element type for source operand \
                of extract operation"
            );
            exit(ExitCode::DialectError);
        }
        let rank_source = s_source.rank().unwrap_or(-1);
        let n_indices = indices.len() as i64;
        if rank_source != n_indices {
            eprintln!("Expected matching arity of indices ({}) and rank ({}) for source operand \
                of extract operation",
                n_indices,
                rank_source,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Extract.get_name(),
        ));
        let mut args = vec![source.clone()];
        args.append(&mut indices.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
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

impl ExtractSlice {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: &RankedTensor,
        source: &Value,
        offsets: &[Value],
        sizes: &[Value],
        strides: &[Value],
        static_offsets: &StaticOffsets,
        static_sizes: &StaticSizes,
        static_strides: &StaticStrides,
        loc: &Location,
    ) -> Self {
        if !source.get_type().is_ranked_tensor() {
            eprintln!("Expected ranked tensor type for source operand of extract slice operation");
            exit(ExitCode::DialectError);
        }
        if offsets.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for offsets of extract slice operation");
            exit(ExitCode::DialectError);
        }
        if sizes.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for sizes of extract slice operation");
            exit(ExitCode::DialectError);
        }
        if strides.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for strides of extract slice operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_source = Shaped::from(*source.get_type().get());
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!("Expected matching result type and element type for source operand \
                of extract slice operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::ExtractSlice.get_name(),
        ));
        let mut args = vec![source.clone()];
        args.append(&mut offsets.to_vec());
        args.append(&mut sizes.to_vec());
        args.append(&mut strides.to_vec());
        let opseg_attr = OperandSegmentSizes::new(&context, &[
            1,
            offsets.len() as i32,
            sizes.len() as i32,
            strides.len() as i32,
        ]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            opseg_attr.as_named_attribute(),
            static_offsets.as_named_attribute(),
            static_sizes.as_named_attribute(),
            static_strides.as_named_attribute(),
        ]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        ExtractSlice(op)
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

    pub fn get_static_offsets(&self) -> StaticOffsets {
        let attr_name = StringBacked::from(StaticOffsets::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticOffsets::from(*attr.get())
    }

    pub fn get_static_sizes(&self) -> StaticSizes {
        let attr_name = StringBacked::from(StaticSizes::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticSizes::from(*attr.get())
    }

    pub fn get_static_strides(&self) -> StaticStrides {
        let attr_name = StringBacked::from(StaticStrides::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticStrides::from(*attr.get())
    }
}

impl FromElements {
    pub fn new(t: &RankedTensor, elements: &[Value], loc: &Location) -> Self {
        let s = t.as_shaped();
        if !s.is_static() {
            eprintln!("Expected static tensor result type of from elements operation");
            exit(ExitCode::DialectError);
        }
        if !s.has_rank() || elements.is_empty() {
            eprintln!("Expected non-empty result tensor type and/or elements array of \
                from elements operation"
            );
            exit(ExitCode::DialectError);
        }
        let n_elem = s.num_elements().unwrap_or(-1);
        let n_elem_inputs = elements.len() as i64;
        if n_elem != n_elem_inputs {
            eprintln!("Expected matching elements size ({}) and inputs ({}) for tensor result type \
                of from elements operation",
                n_elem,
                n_elem_inputs,
            );
            exit(ExitCode::DialectError);
        }
        let t_element = elements.first().unwrap().get_type();
        if elements.iter().any(|v| t_element != v.get_type()) {
            eprintln!("Expected same type for elements in array of from elements operation");
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::FromElements.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(elements);
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

    pub fn get_size_at_index(&self, i: isize) -> Value {
        self.as_operation().get_operand(i)
    }
}

impl Generate {
    pub fn new(t: &RankedTensor, extents: &[Value], loc: &Location) -> Self {
        if extents.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected extents of index type for generate operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let n_dyn_dims = s.num_dynamic_dims().unwrap_or(-1);
        let n_extents = extents.len() as i64;
        if n_dyn_dims != n_extents {
            eprintln!("Expected one index type extent ({}) per dynamic dimension \
                in result tensor type ({}) of generate operation",
                n_extents,
                n_dyn_dims,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Generate.get_name(),
        ));
        let rank = s.rank().unwrap_or(-1) as isize;
        let t_index = Index::new(&context).as_type();
        let t_indices: Vec<Type> = (0..rank).map(|_| t_index.clone()).collect();
        let locs: Vec<Location> = (0..rank).map(|_| loc.clone()).collect();
        let mut region = Region::new();
        let mut block = Block::new(rank, &t_indices, &locs);
        region.append_block(&mut block); // Add empty starter block
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(extents);
        op_state.add_regions(&[region]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Generate(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_block(&self) -> Block {
        self.get_region().iter().next().unwrap()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region(&self) -> Region {
        self.as_operation().iter().next().unwrap()
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Pad {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: &RankedTensor,
        source: &Value,
        values_low: &[Value],
        values_high: &[Value],
        static_low: &StaticLow,
        static_high: &StaticHigh,
        no_fold: Option<NoFold>,
        loc: &Location,
    ) -> Self {
        if !source.get_type().is_ranked_tensor() {
            eprintln!("Expected ranked tensor for source operand of pad operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let t_source = RankedTensor::from(*source.get_type().get());
        let s_source = t_source.as_shaped();
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!("Expected matching element types for source opereand and result type \
                of pad operation"
            );
            exit(ExitCode::DialectError);
        }
        if values_low.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected low values of index type for pad operation");
            exit(ExitCode::DialectError);
        }
        if values_high.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected high values of index type for pad operation");
            exit(ExitCode::DialectError);
        }
        let rank = s.rank().unwrap_or(-1) as isize;
        let n_low = static_low.num_elements() + values_low.len() as isize;
        let n_high = static_high.num_elements() + values_high.len() as isize;
        if rank != n_low {
            eprintln!("Expected arity of low indices ({}) to match rank ({}) for result type \
                of pad operation",
                n_low,
                rank,
            );
            exit(ExitCode::DialectError);
        }
        if rank != n_high {
            eprintln!("Expected arity of high indices ({}) to match rank ({}) for result type \
                of pad operation",
                n_high,
                rank,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Pad.get_name(),
        ));
        let opseg_attr = OperandSegmentSizes::new(&context, &[
            1,
            values_low.len() as i32,
            values_high.len() as i32,
        ]);
        let mut attrs = vec![
            opseg_attr.as_named_attribute(),
            static_low.as_named_attribute(),
            static_high.as_named_attribute(),
        ];
        if let Some(no_fold_) = no_fold {
            attrs.push(no_fold_.as_named_attribute());
        }
        let mut args = vec![source.clone()];
        args.append(&mut values_low.to_vec());
        args.append(&mut values_high.to_vec());
        let t_index = Index::new(&context).as_type();
        let t_indices: Vec<Type> = (0..rank).map(|_| t_index.clone()).collect();
        let locs: Vec<Location> = (0..rank).map(|_| loc.clone()).collect();
        let mut region = Region::new();
        let mut block = Block::new(rank, &t_indices, &locs);
        region.append_block(&mut block); // Add empty starter block
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&attrs);
        op_state.add_operands(&args);
        op_state.add_regions(&[region]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Pad(op)
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

    pub fn get_static_high(&self) -> StaticHigh {
        let attr_name = StringBacked::from(StaticHigh::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticHigh::from(*attr.get())
    }

    pub fn get_static_low(&self) -> StaticLow {
        let attr_name = StringBacked::from(StaticLow::get_name());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        StaticLow::from(*attr.get())
    }

    pub fn is_no_fold(&self) -> bool {
        let attr_name = StringBacked::from(NoFold::get_name());
        self.as_operation().has_attribute_inherent(&attr_name.as_string_ref())
    }
}

impl Rank {
    pub fn new(context: &Context, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_tensor() {
            eprintln!("Expected tensor operand for rank operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Rank.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        op_state.add_results(&[Index::new(context).as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Rank(op)
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

impl Reshape {
    fn new(s: &Shaped, value: &Value, shape: &Value, loc: &Location) -> Self {
        let s_value = Shaped::from_type(&value.get_type());
        if s.get_element_type() != s_value.get_element_type() {
            eprintln!("Expected matching element types for result and source tensor types");
            exit(ExitCode::DialectError);
        }
        let t = s.as_type();
        let context = t.get_context();
        let dialect = context.get_dialect_tensor();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Reshape.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone(), shape.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_ranked(t: &RankedTensor, value: &Value, shape: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        let t_shape = shape.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor for source operand of reshape operation");
            exit(ExitCode::DialectError);
        }
        if !t_shape.is_ranked_tensor() {
            eprintln!("Expected 1D tensor for reshape operation");
            exit(ExitCode::DialectError);
        }
        let s_shape = RankedTensor::from_type(&t_shape).as_shaped();
        if !s_shape.is_static() {
            eprintln!("Expected statically sized shape operand for ranked tensor result \
                of reshape operation"
            );
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_value = Shaped::from_type(&t_value);
        let n_elem = s.num_elements();
        let n_elem_value = s_value.num_elements();
        if s_shape.is_static() && n_elem != n_elem_value {
            eprintln!("Expected matching number of elements for result ({:?}) and source ({:?}) \
                tensor types",
                n_elem,
                n_elem_value,
            );
            exit(ExitCode::DialectError);
        }
        Self::new(&s, value, shape, loc)
    }

    pub fn new_unranked(t: &UnrankedTensor, value: &Value, shape: &Value, loc: &Location) -> Self {
        let t_value = value.get_type();
        let t_shape = shape.get_type();
        if !t_value.is_tensor() {
            eprintln!("Expected tensor for source operand of reshape operation");
            exit(ExitCode::DialectError);
        }
        if !t_shape.is_ranked_tensor() {
            eprintln!("Expected 1D tensor for reshape operation");
            exit(ExitCode::DialectError);
        }
        let s_shape = RankedTensor::from_type(&t_shape).as_shaped();
        if s_shape.is_static() {
            eprintln!("Expected dynamically sized shape operand for unranked tensor result \
                of reshape operation"
            );
            exit(ExitCode::DialectError);
        }
        Self::new(&t.as_shaped(), value, shape, loc)
    }

    pub fn from(op: MlirOperation) -> Self {
        Reshape(op)
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

impl Yield {
    fn __new(
        value: &Value,
        parent: &MlirOperation,
        parent_op: &Op,
        dialect: &Dialect,
        loc: &Location
    ) -> Self {
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Yield.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        Self::from(*op_state.create_operation().get(), *parent, *parent_op)
    }

    pub fn new(value: &Value, parent: &dyn IROperation, loc: &Location) -> Self {
        let context = parent.as_operation().get_context();
        let dialect = context.get_dialect_tensor();
        if parent.get_dialect() != dialect {
            eprintln!("Expected parent operation is from tensor dialect");
            exit(ExitCode::DialectError);
        }
        let parent_op = match parent.get_op().get_name() {
            "generate"  => Op::Generate,
            "pad"       => Op::Pad,
            _           => {
                eprintln!("Expected parent operation is a tensor generate or pad for yield");
                exit(ExitCode::DialectError);
            },
        };
        Self::__new(value, parent.get(), &parent_op, &dialect, loc)
    }

    pub fn new_generate(value: &Value, parent: &Generate, loc: &Location) -> Self {
        Self::__new(value, parent.get(), &Op::Generate, &parent.get_dialect(), loc)
    }

    pub fn new_pad(value: &Value, parent: &Pad, loc: &Location) -> Self {
        Self::__new(value, parent.get(), &Op::Pad, &parent.get_dialect(), loc)
    }

    pub fn from(op: MlirOperation, parent: MlirOperation, parent_op: Op) -> Self {
        Yield(op, parent, parent_op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_parent(&self) -> &MlirOperation {
        &self.1
    }

    pub fn get_parent_op(&self) -> &Op {
        &self.2
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

///////////////////////////////
//  Trait Implemention
///////////////////////////////

impl IROperation for Bitcast {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Dim.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Dim
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for Cast {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Cast.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Cast
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for Concat {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Concat.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Concat
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for CollapseShape {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CollapseShape.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::CollapseShape
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for Dim {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::ShapedDimOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Dim.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Dim
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl IROperation for Empty {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Empty.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Empty
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for ExpandShape {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ExpandShape.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::ExpandShape
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl IROperation for Extract {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
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
        ]
    }
}

impl IROperation for ExtractSlice {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OffsetSizeAndStrideOpInterface,
            Interface::OpAsmOpInterface,
            Interface::ReifyRankedShapeTypeOpInterface,
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
            Trait::AttrSizedOperandSegments,
        ]
    }
}

impl IROperation for FromElements {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
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

impl From<MlirAttribute> for GatherDimensions {
    fn from(attr: MlirAttribute) -> Self {
        GatherDimensions(attr)
    }
}

impl IRAttribute for GatherDimensions {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for GatherDimensions {
    fn get_name() -> &'static str {
        "gather_dims"
    }
}

impl NamedI64DenseArray for GatherDimensions {}

impl IROperation for Generate {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::OpAsmOpInterface,
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Generate.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Generate
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<MlirAttribute> for InnerDimensionsPosition {
    fn from(attr: MlirAttribute) -> Self {
        InnerDimensionsPosition(attr)
    }
}

impl IRAttribute for InnerDimensionsPosition {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for InnerDimensionsPosition {
    fn get_name() -> &'static str {
        "inner_dims_pos"
    }
}

impl NamedI64DenseArray for InnerDimensionsPosition {}

impl From<MlirAttribute> for NoFold {
    fn from(attr: MlirAttribute) -> Self {
        NoFold(attr)
    }
}

impl IRAttribute for NoFold {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for NoFold {
    fn get_name() -> &'static str {
        "nofold"
    }
}

impl NamedUnit for NoFold {}

impl IROp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirAttribute> for OuterDimensionsPermutation {
    fn from(attr: MlirAttribute) -> Self {
        OuterDimensionsPermutation(attr)
    }
}

impl IRAttribute for OuterDimensionsPermutation {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for OuterDimensionsPermutation {
    fn get_name() -> &'static str {
        "outer_dims_perm"
    }
}

impl NamedI64DenseArray for OuterDimensionsPermutation {}

impl IROperation for Pad {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Pad.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Pad
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::AttrSizedOperandSegments,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl IROperation for Rank {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[
            MEFF_NO_MEMORY_EFFECT,
        ]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Rank.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Rank
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl From<MlirAttribute> for Reassociation {
    fn from(attr: MlirAttribute) -> Self {
        Reassociation(attr)
    }
}

impl IRAttribute for Reassociation {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Reassociation {
    fn get_name() -> &'static str {
        "reassociation"
    }
}

impl NamedArrayOfIntegerArrays for Reassociation {}

impl IROperation for Reshape {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Reshape.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Reshape
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
        ]
    }
}

impl From<MlirAttribute> for ScatterDimensions {
    fn from(attr: MlirAttribute) -> Self {
        ScatterDimensions(attr)
    }
}

impl IRAttribute for ScatterDimensions {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for ScatterDimensions {
    fn get_name() -> &'static str {
        "scatter_dims"
    }
}

impl NamedI64DenseArray for ScatterDimensions {}

impl From<MlirAttribute> for StaticHigh {
    fn from(attr: MlirAttribute) -> Self {
        StaticHigh(attr)
    }
}

impl IRAttribute for StaticHigh {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticHigh {
    fn get_name() -> &'static str {
        "static_high"
    }
}

impl NamedI64DenseArray for StaticHigh {}

impl From<MlirAttribute> for StaticInnerTiles {
    fn from(attr: MlirAttribute) -> Self {
        StaticInnerTiles(attr)
    }
}

impl IRAttribute for StaticInnerTiles {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticInnerTiles {
    fn get_name() -> &'static str {
        "static_inner_tiles"
    }
}

impl NamedI64DenseArray for StaticInnerTiles {}

impl From<MlirAttribute> for StaticOutputShape {
    fn from(attr: MlirAttribute) -> Self {
        StaticOutputShape(attr)
    }
}

impl IRAttribute for StaticOutputShape {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticOutputShape {
    fn get_name() -> &'static str {
        "static_output_shape"
    }
}

impl NamedI64DenseArray for StaticOutputShape {}

impl From<MlirAttribute> for StaticLow {
    fn from(attr: MlirAttribute) -> Self {
        StaticLow(attr)
    }
}

impl IRAttribute for StaticLow {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for StaticLow {
    fn get_name() -> &'static str {
        "static_low"
    }
}

impl NamedI64DenseArray for StaticLow {}

impl From<MlirAttribute> for Unique {
    fn from(attr: MlirAttribute) -> Self {
        Unique(attr)
    }
}

impl IRAttribute for Unique {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for Unique {
    fn get_name() -> &'static str {
        "unique"
    }
}

impl NamedUnit for Unique {}

impl IROperation for Yield {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_tensor()
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
            Interface::RegionBranchTerminatorOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Yield.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Yield
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::HasParent(&[&Op::Generate, &Op::Pad]),
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

impl cmp::PartialEq for Yield {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() &&
            self.get_parent_op() == rhs.get_parent_op() &&
            Operation::from(*self.get_parent()) == Operation::from(*rhs.get_parent())
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Bitcast             => "BitCastOp",
            Op::Cast                => "CastOp",
            Op::CollapseShape       => "CollapseShapeOp",
            Op::Concat              => "ConcatOp",
            Op::Dim                 => "DimOp",
            Op::Empty               => "EmptyOp",
            Op::ExpandShape         => "ExpandShapeOp",
            Op::Extract             => "ExtractOp",
            Op::ExtractSlice        => "ExtractSliceOp",
            Op::FromElements        => "FromElementsOp",
            Op::Gather              => "GatherOp",
            Op::Generate            => "GenerateOp",
            Op::Insert              => "InsertOp",
            Op::InsertSlice         => "InsertSliceOp",
            Op::Pack                => "PackOp",
            Op::Pad                 => "PadOp",
            Op::ParallelInsertSlice => "ParallelInsertSliceOp",
            Op::Rank                => "RankOp",
            Op::Reshape             => "ReshapeOp",
            Op::Scatter             => "ScatterOp",
            Op::Splat               => "SplatOp",
            Op::Unpack              => "UnpackOp",
            Op::Yield               => "YieldOp",
        })
    }
}
