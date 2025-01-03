// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]
#![allow(unused_variables)] // TODO: Remove.

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
use attributes::specialized::CustomAttributeData;
use attributes::specialized::NamedArrayOfAffineMaps;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedParsed;
use attributes::specialized::SpecializedAttribute;
use dialects::IOp;
use dialects::IOperation;
use dialects::affine::Dim as AffineDim;
use dialects::affine::IExpr;
use dialects::affine::Map as AffineMap;
use dialects::arith;
use dialects::common::Dimension;
use dialects::common::OperandSegmentSizes;
use effects::MEFF_NO_MEMORY_EFFECT;
use effects::MemoryEffectList;
use exit_code::ExitCode;
use exit_code::exit;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Block;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::Operation;
use ir::OperationState;
use ir::Region;
use ir::Shape;
use ir::ShapeUnpacked;
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::IType;
use types::IsPromotableTo;
use types::ranked_tensor::RankedTensor;
use types::shaped::Shaped;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct BinaryFunction(MlirAttribute);

#[derive(Clone)]
pub struct Cast(MlirAttribute);

#[derive(Clone)]
pub struct IndexingMaps(MlirAttribute);

#[derive(Clone)]
pub struct IteratorType(MlirAttribute);

#[derive(Clone)]
pub struct Permutation(MlirAttribute);

#[derive(Clone)]
pub struct UnaryFunction(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum BinaryFunctionKind {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    DivUnsigned = 4,
    MaxSigned = 5,
    MinSigned = 6,
    MaxUnsigned = 7,
    MinUnsigned = 8,
    PowF = 9,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum CastKind {
    Signed = 0,
    Unsigned = 1,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum IteratorTypeKind {
    Parallel = 0,
    Reduction = 1,
    Window = 2,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Abs,
    Add,
    BatchMatmul,
    BatchMatmulTransposeA,
    BatchMatmulTransposeB,
    BatchMatvec,
    BatchMmt4D,
    BatchReduceMatmul,
    BatchVecmat,
    Broadcast,
    Ceil,
    Conv1DNcwFcw,
    Conv1DNwcWcf,
    Conv1D,
    Conv2D,
    Conv2DNchwFchw,
    Conv2DNgchwGfchw,
    Conv2DNgchwGfchwG,
    Conv2DNhwcFhwc,
    Conv2DNhwcFhwcQ,
    Conv2DNhwcHwcf,
    Conv2DNhwcHwcfQ,
    Conv3D,
    Conv3DNcdhwFcdhw,
    Conv3DNdhwcDhwcf,
    Conv3DNdhwcDhwcfQ,
    Copy,
    DepthwiseConv1DNcwCw,
    DepthwiseConv1DNwcWc,
    DepthwiseConv1DNwcWcm,
    DepthwiseConv2DNchwChw,
    DepthwiseConv2DNhwcHwc,
    DepthwiseConv2DNhwcHwcQ,
    DepthwiseConv2DNhwcHwcm,
    DepthwiseConv2DNhwcHwcmQ,
    DepthwiseConv3DNcdhwCdhw,
    DepthwiseConv3DNdhwcDhwc,
    DepthwiseConv3DNdhwcDhwcm,
    Div,
    DivUnsigned,
    Dot,
    ElementwiseBinary,
    ElementwiseUnary,
    Erf,
    Exp,
    Fill,
    FillRng2D,
    Floor,
    Generic,
    Index,
    Log,
    Map,
    Matmul,
    MatmulTransposeA,
    MatmulTransposeB,
    Matvec,
    Max,
    Min,
    Mmt4D,
    Mul,
    NegF,
    PoolingNchwMax,
    PoolingNchwSum,
    PoolingNcwMax,
    PoolingNcwSum,
    PoolingNdhwcMax,
    PoolingNdhwcMin,
    PoolingNdhwcSum,
    PoolingNhwcMax,
    PoolingNhwcMaxUnsigned,
    PoolingNhwcMin,
    PoolingNhwcMinUnsigned,
    PoolingNhwcSum,
    PoolingNwcMax,
    PoolingNwcMaxUnsigned,
    PoolingNwcMin,
    PoolingNwcMinUnsigned,
    PoolingNwcSum,
    PowF,
    QuantizedBatchMatmul,
    QuantizedMatmul,
    Reciprocal,
    Reduce,
    Round,
    Rsqrt,
    Select,
    Softmax,
    Sqrt,
    Square,
    Sub,
    Tanh,
    Transpose,
    Vecmat,
    WinogradFilterTransform,
    WinogradInputTransform,
    WinogradOutputTransform,
    Yield,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum UnaryFunctionKind {
    Exp = 0,
    Log = 1,
    Abs = 2,
    Ceil = 3,
    Floor = 4,
    NegF = 5,
    Reciprocal = 6,
    Round = 7,
    Sqrt = 8,
    Rsqrt = 9,
    Square = 10,
    Tanh = 11,
    Erf = 12,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Abs(MlirOperation);

#[derive(Clone)]
pub struct Add(MlirOperation);

#[derive(Clone)]
pub struct Ceil(MlirOperation);

#[derive(Clone)]
pub struct Copy(MlirOperation);

#[derive(Clone)]
pub struct Div(MlirOperation);

#[derive(Clone)]
pub struct DivUnsigned(MlirOperation);

#[derive(Clone)]
pub struct Dot(MlirOperation);

#[derive(Clone)]
pub struct ElementwiseBinary(MlirOperation);

#[derive(Clone)]
pub struct ElementwiseUnary(MlirOperation);

#[derive(Clone)]
pub struct Exp(MlirOperation);

#[derive(Clone)]
pub struct Erf(MlirOperation);

#[derive(Clone)]
pub struct Floor(MlirOperation);

#[derive(Clone)]
pub struct Index(MlirOperation);

#[derive(Clone)]
pub struct Matmul(MlirOperation);

#[derive(Clone)]
pub struct MatmulTransposeA(MlirOperation);

#[derive(Clone)]
pub struct MatmulTransposeB(MlirOperation);

#[derive(Clone)]
pub struct Matvec(MlirOperation);

#[derive(Clone)]
pub struct Max(MlirOperation);

#[derive(Clone)]
pub struct Min(MlirOperation);

#[derive(Clone)]
pub struct Mul(MlirOperation);

#[derive(Clone)]
pub struct NegF(MlirOperation);

#[derive(Clone)]
pub struct Log(MlirOperation);

#[derive(Clone)]
pub struct Reciprocal(MlirOperation);

#[derive(Clone)]
pub struct Round(MlirOperation);

#[derive(Clone)]
pub struct Rsqrt(MlirOperation);

#[derive(Clone)]
pub struct Sqrt(MlirOperation);

#[derive(Clone)]
pub struct Square(MlirOperation);

#[derive(Clone)]
pub struct Sub(MlirOperation);

#[derive(Clone)]
pub struct Tanh(MlirOperation);

#[derive(Clone)]
pub struct Transpose(MlirOperation);

#[derive(Clone)]
pub struct Vecmat(MlirOperation);

#[derive(Clone)]
pub struct Yield(MlirOperation);

///////////////////////////////
//  Traits
///////////////////////////////

pub trait TransformShape {
    fn matmul(&self, rhs: &Self) -> Option<Vec<i64>>;
    fn matmul_transpose_a(&self, rhs: &Self) -> Option<Vec<i64>>;
    fn matmul_transpose_b(&self, rhs: &Self) -> Option<Vec<i64>>;
    fn matvec(&self, rhs: &Self) -> Option<Vec<i64>>;
    fn vecmat(&self, rhs: &Self) -> Option<Vec<i64>>;
    fn unpack_matmul(&self, rhs: &Self) -> Option<ShapeUnpacked>;
    fn unpack_matmul_transpose_a(&self, rhs: &Self) -> Option<ShapeUnpacked>;
    fn unpack_matmul_transpose_b(&self, rhs: &Self) -> Option<ShapeUnpacked>;
    fn unpack_matvec(&self, rhs: &Self) -> Option<ShapeUnpacked>;
    fn unpack_vecmat(&self, rhs: &Self) -> Option<ShapeUnpacked>;
}

pub trait ElementwiseOperation: From<MlirOperation> + IOperation + Sized {
    fn get_result(&self) -> Option<Value> {
        let op = self.as_operation();
        if op.num_results() > 0 {
            Some(op.get_result(0))
        } else {
            None
        }
    }
}

pub trait ElementwiseCheckBinaryOperands {
    fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> ();
}

pub trait ElementwiseCheckUnaryOperands {
    fn __check_operands(op: &'static Op, input: &Value, output: &Value) -> ();
}

pub trait ElementwiseCheckResult {
    fn __check_result(op: &'static Op, t: &RankedTensor, t_out: &Type) -> ();
}

pub trait ElementwiseBinaryOperationGetBody {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation>;
}

pub trait ElementwiseUnaryOperationGetBody {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation>;
}

pub trait ElementwiseBinaryOperation:
    ElementwiseBinaryOperationGetBody
    + ElementwiseCheckBinaryOperands
    + ElementwiseCheckResult
    + ElementwiseOperation
{
    fn __new_body(t: &Type, op_parent: &mut Operation, loc: &Location) -> Self {
        let context = t.get_context();
        let mut block = op_parent.get_region(0).iter().next().unwrap_or_default();
        let lhs = block.get_arg(0);
        let rhs = block.get_arg(1);
        let acc = block.get_arg(2);
        let ops = Self::get_body(t, &lhs, &rhs, &acc, loc);
        let results = if let Some(op) = ops.last() {
            vec![op.get_result(0)]
        } else {
            vec![acc.clone()]
        };
        let mut op_yield = Yield::new(&context, &results, loc).as_operation();
        ops.into_iter()
            .for_each(|mut op| block.append_operation(&mut op));
        block.append_operation(&mut op_yield);
        Self::from(*op_parent.get_mut())
    }

    fn __new_memref(
        op: &'static Op,
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        if !lhs.get_type().is_memref()
            || !rhs.get_type().is_memref()
            || !output.get_type().is_memref()
        {
            eprintln!(
                "Expected memory reference type operand(s) for {} operation",
                op.get_name()
            );
            exit(ExitCode::DialectError);
        }
        Self::__check_operands(op, lhs, rhs, output);
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(op);
        let mut region = Region::new();
        let t_elem = Shaped::from_type(&output.get_type()).get_element_type();
        region.append_block(&mut Block::new(
            3,
            &[t_elem.clone(), t_elem.clone(), t_elem.clone()],
            &[loc.clone(), loc.clone(), loc.clone()],
        ));
        let opseg_attr = OperandSegmentSizes::new(context, &[2, 1]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone(), output.clone()]);
        op_state.add_regions(&[region]);
        Self::__new_body(&t_elem, &mut op_state.create_operation(), loc)
    }

    fn __new_tensor(
        op: &'static Op,
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        if !lhs.get_type().is_tensor()
            || !rhs.get_type().is_tensor()
            || !output.get_type().is_tensor()
        {
            eprintln!(
                "Expected tensor type operand(s) for {} operation",
                op.get_name()
            );
            exit(ExitCode::DialectError);
        }
        Self::__check_operands(op, lhs, rhs, output);
        Self::__check_result(op, t, &output.get_type());
        let context = t.get_context();
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(op);
        let mut region = Region::new();
        let t_elem = t.as_shaped().get_element_type();
        region.append_block(&mut Block::new(
            3,
            &[t_elem.clone(), t_elem.clone(), t_elem.clone()],
            &[loc.clone(), loc.clone(), loc.clone()],
        ));
        let opseg_attr = OperandSegmentSizes::new(&context, &[2, 1]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute()]);
        op_state.add_operands(&[lhs.clone(), rhs.clone(), output.clone()]);
        op_state.add_regions(&[region]);
        op_state.add_results(&[t.as_type()]);
        Self::__new_body(&t_elem, &mut op_state.create_operation(), loc)
    }
}

pub trait ElementwiseUnaryOperation:
    ElementwiseUnaryOperationGetBody
    + ElementwiseCheckUnaryOperands
    + ElementwiseCheckResult
    + ElementwiseOperation
{
    fn __new_body(t: &Type, op_parent: &mut Operation, loc: &Location) -> Self {
        let context = t.get_context();
        let mut block = op_parent.get_region(0).iter().next().unwrap_or_default();
        let input = block.get_arg(0);
        let acc = block.get_arg(1);
        let ops = Self::get_body(t, &input, &acc, loc);
        let results = if let Some(op) = ops.last() {
            vec![op.get_result(0)]
        } else {
            vec![input.clone()]
        };
        let mut op_yield = Yield::new(&context, &results, loc).as_operation();
        ops.into_iter()
            .for_each(|mut op| block.append_operation(&mut op));
        block.append_operation(&mut op_yield);
        Self::from(*op_parent.get_mut())
    }

    fn __new_memref(
        op: &'static Op,
        context: &Context,
        input: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        if !input.get_type().is_memref() || !output.get_type().is_memref() {
            eprintln!(
                "Expected memory reference type operand(s) for {} operation",
                op.get_name()
            );
            exit(ExitCode::DialectError);
        }
        Self::__check_operands(op, input, output);
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(op);
        let mut region = Region::new();
        let t_elem = Shaped::from_type(&output.get_type()).get_element_type();
        region.append_block(&mut Block::new(2, &[t_elem.clone(), t_elem.clone()], &[
            loc.clone(),
            loc.clone(),
        ]));
        let opseg_attr = OperandSegmentSizes::new(context, &[1, 1]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute()]);
        op_state.add_operands(&[input.clone(), output.clone()]);
        op_state.add_regions(&[region]);
        Self::__new_body(&t_elem, &mut op_state.create_operation(), loc)
    }

    fn __new_tensor(
        op: &'static Op,
        t: &RankedTensor,
        input: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        if !input.get_type().is_tensor() || !output.get_type().is_tensor() {
            eprintln!(
                "Expected tensor type operand(s) for {} operation",
                op.get_name()
            );
            exit(ExitCode::DialectError);
        }
        Self::__check_operands(op, input, output);
        Self::__check_result(op, t, &output.get_type());
        let context = t.get_context();
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(op);
        let mut region = Region::new();
        let t_elem = t.as_shaped().get_element_type();
        region.append_block(&mut Block::new(2, &[t_elem.clone(), t_elem.clone()], &[
            loc.clone(),
            loc.clone(),
        ]));
        let opseg_attr = OperandSegmentSizes::new(&context, &[1, 1]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute()]);
        op_state.add_operands(&[input.clone(), output.clone()]);
        op_state.add_regions(&[region]);
        op_state.add_results(&[t.as_type()]);
        Self::__new_body(&t_elem, &mut op_state.create_operation(), loc)
    }
}

macro_rules! impl_ElementwiseCheckShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckResult for $OperationName {
            fn __check_result(op: &'static Op, t: &RankedTensor, t_out: &Type) -> () {
                let s = t.as_shaped();
                let s_out = Shaped::from(*t_out.get());
                if s.get_element_type() != s_out.get_element_type() {
                    eprintln!(
                        "Expected matching element types for output and result of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s.unpack() != s_out.unpack() {
                    eprintln!(
                        "Expected matching shapes for output and result of {} operation",
                        op.get_name()
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    };
}

macro_rules! impl_ElementwiseCheckScalarResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckResult for $OperationName {
            fn __check_result(op: &'static Op, t: &RankedTensor, t_out: &Type) -> () {
                if t.as_shaped().get_element_type() != *t_out {
                    eprintln!(
                        "Expected matching element type for output and result type of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    };
}

macro_rules! impl_ElementwiseCheckBinaryOperandsMatchingShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked != s_lhs.unpack() || s_unpacked != s_rhs.unpack() {
                    eprintln!(
                        "Expected matching shapes for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckBinaryOperandsMatmulShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                let s_unpacked_matmul = s_lhs.unpack_matmul(&s_rhs);
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                // TODO: Shape check depends on indexing maps.
                //if s_unpacked_matmul.is_none() || s_unpacked != s_unpacked_matmul.unwrap()  {
                //    eprintln!(
                //        "Expected compatible matmul output shape for operands of {} operation",
                //        op.get_name(),
                //    );
                //    exit(ExitCode::DialectError);
                //}
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckBinaryOperandsMatmulTransposeAShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                let s_unpacked_matmul = s_lhs.unpack_matmul_transpose_a(&s_rhs);
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked_matmul.is_none() || s_unpacked != s_unpacked_matmul.unwrap()  {
                    eprintln!(
                        "Expected compatible matmul output shape for operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckBinaryOperandsMatmulTransposeBShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                let s_unpacked_matmul = s_lhs.unpack_matmul_transpose_b(&s_rhs);
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked_matmul.is_none() || s_unpacked != s_unpacked_matmul.unwrap()  {
                    eprintln!(
                        "Expected compatible matmul output shape for operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckBinaryOperandsMatvecShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                let s_unpacked_matvec = s_lhs.unpack_matvec(&s_rhs).unwrap_or_default();
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked != s_unpacked_matvec {
                    eprintln!(
                        "Expected compatible matvec output shape ({:?}) for result shape ({:?}) of \
                        {} operation",
                        s_unpacked,
                        s_unpacked_matvec,
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckBinaryOperandsPromotableScalarResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let t_elem = output.get_type();
                if !s_lhs.get_element_type().is_promotable_to(&t_elem)
                    || !s_rhs.get_element_type().is_promotable_to(&t_elem)
                {
                    eprintln!(
                        "Expected promotable element type for inputs to output operand type \
                        of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_lhs.unpack() != s_rhs.unpack() {
                    eprintln!(
                        "Expected matching shapes for input operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    };
}

macro_rules! impl_ElementwiseCheckBinaryOperandsPromotableShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let s_unpacked = s_output.unpack();
                let t_elem = s_output.get_element_type();
                if !s_lhs.get_element_type().is_promotable_to(&t_elem)
                    || !s_rhs.get_element_type().is_promotable_to(&t_elem)
                {
                    eprintln!(
                        "Expected promotable element type for inputs to output operand type \
                        of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked != s_lhs.unpack() || s_unpacked != s_rhs.unpack() {
                    eprintln!(
                        "Expected matching shapes for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    };
}

macro_rules! impl_ElementwiseCheckBinaryOperandsVecmatShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckBinaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, lhs: &Value, rhs: &Value, output: &Value) -> () {
                let s_lhs = Shaped::from(*lhs.get_type().get());
                let s_rhs = Shaped::from(*rhs.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                let t_elem = s_output.get_element_type();
                let s_unpacked = s_output.unpack();
                let s_unpacked_vecmat = s_lhs.unpack_vecmat(&s_rhs).unwrap_or_default();
                if t_elem != s_lhs.get_element_type() || t_elem != s_rhs.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_unpacked != s_unpacked_vecmat  {
                    eprintln!(
                        "Expected compatible vecmat output shape ({:?}) for result shape ({:?}) of \
                        {} operation",
                        s_unpacked,
                        s_unpacked_vecmat,
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckUnaryOperandsMatchingShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckUnaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, input: &Value, output: &Value) -> () {
                let s_input = Shaped::from(*input.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                if s_output.get_element_type() != s_input.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_output.unpack() != s_input.unpack() {
                    eprintln!(
                        "Expected matching shapes for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseCheckUnaryOperandsPromotableShapedResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckUnaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, input: &Value, output: &Value) -> () {
                let s_input = Shaped::from(*input.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                if !s_input
                    .get_element_type()
                    .is_promotable_to(&s_output.get_element_type())
                {
                    eprintln!(
                        "Expected promotable element type for input to output operand type \
                        of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_output.unpack() != s_input.unpack() {
                    eprintln!(
                        "Expected matching shapes for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    };
}

macro_rules! impl_ElementwiseCheckUnaryOperandsMatchingTransposeResult {
    ($OperationName:ident) => {
        impl ElementwiseCheckUnaryOperands for $OperationName {
            fn __check_operands(op: &'static Op, input: &Value, output: &Value) -> () {
                let s_input = Shaped::from(*input.get_type().get());
                let s_output = Shaped::from(*output.get_type().get());
                if s_output.get_element_type() != s_input.get_element_type() {
                    eprintln!(
                        "Expected matching element types for input and output operands of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
                if s_output.unpack_transpose() != s_input.unpack() {
                    eprintln!(
                        "Expected transposed output shape from input operand of {} operation",
                        op.get_name(),
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
    }
}

macro_rules! impl_ElementwiseBinaryOpPromotableTypeOperandsAndScalarResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckScalarResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsPromotableScalarResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsMatchingShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpMatmulTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsMatmulShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpMatmulTransposeATypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsMatmulTransposeAShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpMatmulTransposeBTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsMatmulTransposeBShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpMatvecTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsMatvecShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpVecmatTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsVecmatShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseBinaryOpPromotableTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckBinaryOperandsPromotableShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseBinaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckUnaryOperandsMatchingShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseUnaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseUnaryOpMatchingTypeOperandsAndTransposeResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckUnaryOperandsMatchingTransposeResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseUnaryOperation for $OperationName {}
    };
}

macro_rules! impl_ElementwiseUnaryOpPromotableTypeOperandsAndShapedResult {
    ($OperationName:ident) => {
        impl_ElementwiseCheckShapedResult!($OperationName);
        impl_ElementwiseCheckUnaryOperandsPromotableShapedResult!($OperationName);
        impl ElementwiseOperation for $OperationName {}
        impl ElementwiseUnaryOperation for $OperationName {}
    };
}

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl BinaryFunction {
    pub fn new(context: &Context, k: BinaryFunctionKind) -> Self {
        const WIDTH: usize = 32;
        <Self as NamedInteger>::new(context, k as i64, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_kind(&self) -> BinaryFunctionKind {
        BinaryFunctionKind::from(self.get_value() as i32)
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Cast {
    pub fn new(context: &Context, k: CastKind) -> Self {
        let cad = CustomAttributeData::new(
            "type_fn".to_string(),
            context.get_dialect_linalg().get_namespace().to_string(),
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
}

impl IndexingMaps {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IteratorType {
    pub fn new(context: &Context, k: IteratorTypeKind) -> Self {
        const WIDTH: usize = 32;
        <Self as NamedInteger>::new(context, k as i64, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_kind(&self) -> IteratorTypeKind {
        IteratorTypeKind::from(self.get_value() as i32)
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Permutation {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl UnaryFunction {
    pub fn new(context: &Context, k: UnaryFunctionKind) -> Self {
        const WIDTH: usize = 32;
        <Self as NamedInteger>::new(context, k as i64, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_kind(&self) -> UnaryFunctionKind {
        UnaryFunctionKind::from(self.get_value() as i32)
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

///////////////////////////////
//  Enum Implementation
///////////////////////////////

impl BinaryFunctionKind {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => BinaryFunctionKind::Add,
            1 => BinaryFunctionKind::Sub,
            2 => BinaryFunctionKind::Mul,
            3 => BinaryFunctionKind::Div,
            4 => BinaryFunctionKind::DivUnsigned,
            5 => BinaryFunctionKind::MaxSigned,
            6 => BinaryFunctionKind::MinSigned,
            7 => BinaryFunctionKind::MaxUnsigned,
            8 => BinaryFunctionKind::MinUnsigned,
            9 => BinaryFunctionKind::PowF,
            _ => {
                eprintln!("Invalid value '{}' for BinaryFunctionKind", n);
                exit(ExitCode::DialectError);
            }
        }
    }
}

impl CastKind {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => CastKind::Signed,
            1 => CastKind::Unsigned,
            _ => {
                eprintln!("Invalid value '{}' for CastKind", n);
                exit(ExitCode::DialectError);
            }
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            CastKind::Signed => "cast_signed",
            CastKind::Unsigned => "cast_unsigned",
        }
    }
}

impl IteratorTypeKind {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => IteratorTypeKind::Parallel,
            1 => IteratorTypeKind::Reduction,
            2 => IteratorTypeKind::Window,
            _ => {
                eprintln!("Invalid value '{}' for IteratorTypeKind", n);
                exit(ExitCode::DialectError);
            }
        }
    }
}

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::Abs => "abs",
            Op::Add => "add",
            Op::BatchMatmul => "batch_matmul",
            Op::BatchMatmulTransposeA => "batch_matmul_transpose_a",
            Op::BatchMatmulTransposeB => "batch_matmul_transpose_b",
            Op::BatchMatvec => "batch_matvec",
            Op::BatchMmt4D => "batch_mmt4d",
            Op::BatchReduceMatmul => "batch_reduce_matmul",
            Op::BatchVecmat => "batch_vecmat",
            Op::Broadcast => "broadcast",
            Op::Ceil => "ceil",
            Op::Conv1DNcwFcw => "conv_1d_ncw_fcw",
            Op::Conv1DNwcWcf => "conv_1d_nwc_wcf",
            Op::Conv1D => "conv_1d",
            Op::Conv2D => "conv_2d",
            Op::Conv2DNchwFchw => "conv_2d_nchw_fchw",
            Op::Conv2DNgchwGfchw => "conv_2d_ngchw_gfchw",
            Op::Conv2DNgchwGfchwG => "conv_2d_ngchw_gfchw_g",
            Op::Conv2DNhwcFhwc => "conv_2d_nhwc_fhwc",
            Op::Conv2DNhwcFhwcQ => "conv_2d_nhwc_fhwc_q",
            Op::Conv2DNhwcHwcf => "conv_2d_nhwc_hwcf",
            Op::Conv2DNhwcHwcfQ => "conv_2d_nhwc_hwcf_q",
            Op::Conv3D => "conv_3d",
            Op::Conv3DNcdhwFcdhw => "conv_3d_ncdhw_fcdhw",
            Op::Conv3DNdhwcDhwcf => "conv_3d_ndhwc_dhwcf",
            Op::Conv3DNdhwcDhwcfQ => "conv_3d_ndhwc_dhwcf_q",
            Op::Copy => "copy",
            Op::DepthwiseConv1DNcwCw => "depthwise_conv1d_ncw_cw",
            Op::DepthwiseConv1DNwcWc => "depthwise_conv1d_nwc_wc",
            Op::DepthwiseConv1DNwcWcm => "depthwise_conv1d_nwc_wcm",
            Op::DepthwiseConv2DNchwChw => "depthwise_conv2d_nchw_chw",
            Op::DepthwiseConv2DNhwcHwc => "depthwise_conv2d_nhwc_hwc",
            Op::DepthwiseConv2DNhwcHwcQ => "depthwise_conv2d_nhwc_hwc_q",
            Op::DepthwiseConv2DNhwcHwcm => "depthwise_conv2d_nhwc_hwcm",
            Op::DepthwiseConv2DNhwcHwcmQ => "depthwise_conv2d_nhwc_hwcm_q",
            Op::DepthwiseConv3DNcdhwCdhw => "depthwise_conv3d_ncdhw_cdhw",
            Op::DepthwiseConv3DNdhwcDhwc => "depthwise_conv3d_ndhwc_dhwc",
            Op::DepthwiseConv3DNdhwcDhwcm => "depthwise_conv3d_ndhwc_dhwcm",
            Op::Div => "div",
            Op::DivUnsigned => "div_unsigned",
            Op::Dot => "dot",
            Op::ElementwiseBinary => "elementwise_binary",
            Op::ElementwiseUnary => "elementwise_unary",
            Op::Erf => "erf",
            Op::Exp => "exp",
            Op::Fill => "fill",
            Op::FillRng2D => "fill_rng_2d",
            Op::Floor => "floor",
            Op::Generic => "generic",
            Op::Index => "index",
            Op::Log => "log",
            Op::Map => "map",
            Op::Matmul => "matmul",
            Op::MatmulTransposeA => "matmul_transpose_a",
            Op::MatmulTransposeB => "matmul_transpose_b",
            Op::Matvec => "matvec",
            Op::Max => "max",
            Op::Min => "min",
            Op::Mmt4D => "mmt4d",
            Op::Mul => "mul",
            Op::NegF => "negf",
            Op::PoolingNchwMax => "pooling_nchw_max",
            Op::PoolingNchwSum => "pooling_nchw_sum",
            Op::PoolingNcwMax => "pooling_ncw_max",
            Op::PoolingNcwSum => "pooling_ncw_sum",
            Op::PoolingNdhwcMax => "pooling_ndhwc_max",
            Op::PoolingNdhwcMin => "pooling_ndhwc_min",
            Op::PoolingNdhwcSum => "pooling_ndhwc_sum",
            Op::PoolingNhwcMax => "pooling_nhwc_max",
            Op::PoolingNhwcMaxUnsigned => "pooling_nhwc_max_unsigned",
            Op::PoolingNhwcMin => "pooling_nhwc_min",
            Op::PoolingNhwcMinUnsigned => "pooling_nhwc_min_unsigned",
            Op::PoolingNhwcSum => "pooling_nhwc_sum",
            Op::PoolingNwcMax => "pooling_nwc_max",
            Op::PoolingNwcMaxUnsigned => "pooling_nwc_max_unsigned",
            Op::PoolingNwcMin => "pooling_nwc_min",
            Op::PoolingNwcMinUnsigned => "pooling_nwc_min_unsigned",
            Op::PoolingNwcSum => "pooling_nwc_sum",
            Op::PowF => "powf",
            Op::QuantizedBatchMatmul => "quantized_batch_matmul",
            Op::QuantizedMatmul => "quantized_matmul",
            Op::Reciprocal => "reciprocal",
            Op::Reduce => "reduce",
            Op::Round => "round",
            Op::Rsqrt => "rsqrt",
            Op::Select => "select",
            Op::Softmax => "softmax",
            Op::Sqrt => "sqrt",
            Op::Square => "square",
            Op::Sub => "sub",
            Op::Tanh => "tanh",
            Op::Transpose => "transpose",
            Op::Vecmat => "vecmat",
            Op::WinogradFilterTransform => "winograd_filter_transform",
            Op::WinogradInputTransform => "winograd_input_transform",
            Op::WinogradOutputTransform => "winograd_output_transform",
            Op::Yield => "yield",
        }
    }
}

impl UnaryFunctionKind {
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => UnaryFunctionKind::Exp,
            1 => UnaryFunctionKind::Log,
            2 => UnaryFunctionKind::Abs,
            3 => UnaryFunctionKind::Ceil,
            4 => UnaryFunctionKind::Floor,
            5 => UnaryFunctionKind::NegF,
            6 => UnaryFunctionKind::Reciprocal,
            7 => UnaryFunctionKind::Round,
            8 => UnaryFunctionKind::Sqrt,
            9 => UnaryFunctionKind::Rsqrt,
            10 => UnaryFunctionKind::Square,
            11 => UnaryFunctionKind::Tanh,
            12 => UnaryFunctionKind::Erf,
            _ => {
                eprintln!("Invalid value '{}' for UnaryFunctionKind", n);
                exit(ExitCode::DialectError);
            }
        }
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

impl Abs {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Abs, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Abs, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Abs, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Abs, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Add {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Add, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Add, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Add, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Add, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Ceil {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Ceil, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Ceil, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Ceil, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Ceil, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Copy {
    pub fn new_memref(
        context: &Context,
        input: &Value,
        output: &Value,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op = Self::__new_memref(&Op::Copy, context, input, output, loc).as_operation();
        let cast = Cast::new(context, cast_kind).as_named_attribute();
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    pub fn new_tensor(
        t: &RankedTensor,
        input: &Value,
        output: &Value,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op = Self::__new_tensor(&Op::Copy, t, input, output, loc).as_operation();
        let cast = Cast::new(&t.get_context(), cast_kind).as_named_attribute();
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Copy, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Copy, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cast_kind(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Div {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Div, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Div, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Div, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Div, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl DivUnsigned {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::DivUnsigned, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::DivUnsigned, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::DivUnsigned, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::DivUnsigned, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Dot {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Dot, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Dot, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Dot, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Dot, t, t_out)
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

impl ElementwiseBinary {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        f_kind: BinaryFunctionKind,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op = Self::__new_memref(&Op::ElementwiseBinary, context, lhs, rhs, output, loc)
            .as_operation();
        let f = BinaryFunction::new(context, f_kind).as_named_attribute();
        let cast = Cast::new(context, cast_kind).as_named_attribute();
        op.set_attribute_inherent(&f.get_identifier().as_string(), &f.as_attribute());
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        f_kind: BinaryFunctionKind,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op =
            Self::__new_tensor(&Op::ElementwiseBinary, t, lhs, rhs, output, loc).as_operation();
        let f = BinaryFunction::new(&t.get_context(), f_kind).as_named_attribute();
        let cast = Cast::new(&t.get_context(), cast_kind).as_named_attribute();
        op.set_attribute_inherent(&f.get_identifier().as_string(), &f.as_attribute());
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::ElementwiseBinary, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::ElementwiseBinary, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_binary_function(&self) -> BinaryFunction {
        let attr_name = StringBacked::from(BinaryFunction::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        BinaryFunction::from(*attr.get())
    }

    pub fn get_cast_kind(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl ElementwiseUnary {
    pub fn new_memref(
        context: &Context,
        input: &Value,
        output: &Value,
        f_kind: UnaryFunctionKind,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op =
            Self::__new_memref(&Op::ElementwiseUnary, context, input, output, loc).as_operation();
        let f = UnaryFunction::new(context, f_kind).as_named_attribute();
        let cast = Cast::new(context, cast_kind).as_named_attribute();
        op.set_attribute_inherent(&f.get_identifier().as_string(), &f.as_attribute());
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    pub fn new_tensor(
        t: &RankedTensor,
        input: &Value,
        output: &Value,
        f_kind: UnaryFunctionKind,
        cast_kind: CastKind,
        loc: &Location,
    ) -> Self {
        let mut op =
            Self::__new_tensor(&Op::ElementwiseUnary, t, input, output, loc).as_operation();
        let f = UnaryFunction::new(&t.get_context(), f_kind).as_named_attribute();
        let cast = Cast::new(&t.get_context(), cast_kind).as_named_attribute();
        op.set_attribute_inherent(&f.get_identifier().as_string(), &f.as_attribute());
        op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        Self::from(*op.get_mut())
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::ElementwiseUnary, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::ElementwiseUnary, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cast_kind(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_unary_function(&self) -> UnaryFunction {
        let attr_name = StringBacked::from(UnaryFunction::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        UnaryFunction::from(*attr.get())
    }
}

impl Erf {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Erf, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Erf, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Erf, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Erf, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Exp {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Exp, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Exp, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Exp, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Exp, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Floor {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Floor, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Floor, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Floor, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Floor, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Index {
    pub fn new(context: &Context, dim: &Dimension, loc: &Location) -> Self {
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(&Op::Index);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[dim.as_named_attribute()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_dimension(&self) -> Dimension {
        let attr_name = StringBacked::from(Dimension::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Dimension::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Log {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Log, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Log, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Log, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Log, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Matmul {
    fn new(
        op: &mut Operation,
        index_maps: Option<&IndexingMaps>,
        cast_kind: Option<CastKind>,
    ) -> Self {
        let context = op.get_context();
        let index_maps_ = index_maps
            .unwrap_or(&Self::get_default_indexing_maps(&context))
            .as_named_attribute();
        op.set_attribute_discardable(
            &index_maps_.get_identifier().as_string(),
            &index_maps_.as_attribute(),
        );
        if let Some(cast_kind_) = cast_kind {
            let cast = Cast::new(&context, cast_kind_).as_named_attribute();
            op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        }
        Self::from(*op.get_mut())
    }

    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        index_maps: Option<&IndexingMaps>,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_memref(&Op::Matmul, context, lhs, rhs, output, loc).as_operation(),
            index_maps,
            cast_kind,
        )
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        index_maps: Option<&IndexingMaps>,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_tensor(&Op::Matmul, t, lhs, rhs, output, loc).as_operation(),
            index_maps,
            cast_kind,
        )
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Matmul, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Matmul, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cast(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_default_indexing_maps(context: &Context) -> IndexingMaps {
        let d0 = AffineDim::new(context, 0).as_expr();
        let d1 = AffineDim::new(context, 1).as_expr();
        let d2 = AffineDim::new(context, 2).as_expr();
        let map0 = AffineMap::new_results(context, 3, 0, &[d0, d2]);
        let map1 = AffineMap::new_results(context, 3, 0, &[d2, d1]);
        let map2 = AffineMap::new_results(context, 3, 0, &[d0, d1]);
        IndexingMaps::new(context, &[map0, map1, map2])
    }

    pub fn get_indexing_maps(&self) -> IndexingMaps {
        let attr_name = StringBacked::from(IndexingMaps::get_name());
        let attr = self
            .as_operation()
            .get_attribute_discardable(&attr_name.as_string_ref());
        IndexingMaps::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl MatmulTransposeA {
    fn new(op: &mut Operation, cast_kind: Option<CastKind>) -> Self {
        let context = op.get_context();
        let index_maps = Self::get_default_indexing_maps(&context).as_named_attribute();
        op.set_attribute_discardable(
            &index_maps.get_identifier().as_string(),
            &index_maps.as_attribute(),
        );
        if let Some(cast_kind_) = cast_kind {
            let cast = Cast::new(&context, cast_kind_).as_named_attribute();
            op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        }
        Self::from(*op.get_mut())
    }

    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_memref(&Op::MatmulTransposeA, context, lhs, rhs, output, loc)
                .as_operation(),
            cast_kind,
        )
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_tensor(&Op::MatmulTransposeA, t, lhs, rhs, output, loc).as_operation(),
            cast_kind,
        )
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::MatmulTransposeA, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::MatmulTransposeA, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cast(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_default_indexing_maps(context: &Context) -> IndexingMaps {
        let d0 = AffineDim::new(context, 0).as_expr();
        let d1 = AffineDim::new(context, 1).as_expr();
        let d2 = AffineDim::new(context, 2).as_expr();
        let map0 = AffineMap::new_results(context, 3, 0, &[d2, d0]);
        let map1 = AffineMap::new_results(context, 3, 0, &[d2, d1]);
        let map2 = AffineMap::new_results(context, 3, 0, &[d0, d1]);
        IndexingMaps::new(context, &[map0, map1, map2])
    }

    pub fn get_indexing_maps(&self) -> IndexingMaps {
        let attr_name = StringBacked::from(IndexingMaps::get_name());
        let attr = self
            .as_operation()
            .get_attribute_discardable(&attr_name.as_string_ref());
        IndexingMaps::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl MatmulTransposeB {
    pub fn new(op: &mut Operation, cast_kind: Option<CastKind>) -> Self {
        let context = op.get_context();
        let index_maps = Self::get_default_indexing_maps(&context).as_named_attribute();
        op.set_attribute_discardable(
            &index_maps.get_identifier().as_string(),
            &index_maps.as_attribute(),
        );
        if let Some(cast_kind_) = cast_kind {
            let cast = Cast::new(&context, cast_kind_).as_named_attribute();
            op.set_attribute_inherent(&cast.get_identifier().as_string(), &cast.as_attribute());
        }
        Self::from(*op.get_mut())
    }

    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_memref(&Op::MatmulTransposeB, context, lhs, rhs, output, loc)
                .as_operation(),
            cast_kind,
        )
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        cast_kind: Option<CastKind>,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_tensor(&Op::MatmulTransposeB, t, lhs, rhs, output, loc).as_operation(),
            cast_kind,
        )
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::MatmulTransposeB, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::MatmulTransposeB, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cast(&self) -> Cast {
        let attr_name = StringBacked::from(Cast::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cast::from(*attr.get())
    }

    pub fn get_default_indexing_maps(context: &Context) -> IndexingMaps {
        let d0 = AffineDim::new(context, 0).as_expr();
        let d1 = AffineDim::new(context, 1).as_expr();
        let d2 = AffineDim::new(context, 2).as_expr();
        let map0 = AffineMap::new_results(context, 3, 0, &[d0, d2]);
        let map1 = AffineMap::new_results(context, 3, 0, &[d1, d2]);
        let map2 = AffineMap::new_results(context, 3, 0, &[d0, d1]);
        IndexingMaps::new(context, &[map0, map1, map2])
    }

    pub fn get_indexing_maps(&self) -> IndexingMaps {
        let attr_name = StringBacked::from(IndexingMaps::get_name());
        let attr = self
            .as_operation()
            .get_attribute_discardable(&attr_name.as_string_ref());
        IndexingMaps::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Matvec {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Matvec, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Matvec, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Matvec, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Matvec, t, t_out)
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

impl Max {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Max, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Max, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Max, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Max, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Min {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Min, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Min, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Min, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Min, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Mul {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Mul, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Mul, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Mul, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Mul, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl NegF {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::NegF, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::NegF, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::NegF, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::NegF, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Reciprocal {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Reciprocal, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Reciprocal, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Reciprocal, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Reciprocal, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Round {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Round, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Round, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Round, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Round, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Rsqrt {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Rsqrt, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Rsqrt, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Rsqrt, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Rsqrt, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Sqrt {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Sqrt, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Sqrt, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Sqrt, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Sqrt, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Square {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Square, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Square, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Square, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Square, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Sub {
    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_memref(&Op::Sub, context, lhs, rhs, output, loc)
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new_tensor(&Op::Sub, t, lhs, rhs, output, loc)
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Sub, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Sub, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Tanh {
    pub fn new_memref(context: &Context, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_memref(&Op::Tanh, context, input, output, loc)
    }

    pub fn new_tensor(t: &RankedTensor, input: &Value, output: &Value, loc: &Location) -> Self {
        Self::__new_tensor(&Op::Tanh, t, input, output, loc)
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Tanh, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Tanh, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Transpose {
    fn new(op: &mut Operation, p: &Permutation) -> Self {
        let p_ = p.as_named_attribute();
        op.set_attribute_inherent(&p_.get_identifier().as_string(), &p_.as_attribute());
        Self::from(*op.get_mut())
    }

    pub fn new_memref(
        context: &Context,
        input: &Value,
        output: &Value,
        p: &Permutation,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_memref(&Op::Transpose, context, input, output, loc).as_operation(),
            p,
        )
    }

    pub fn new_tensor(
        t: &RankedTensor,
        input: &Value,
        output: &Value,
        p: &Permutation,
        loc: &Location,
    ) -> Self {
        Self::new(
            &mut Self::__new_tensor(&Op::Transpose, t, input, output, loc).as_operation(),
            p,
        )
    }

    fn check_operands(input: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Transpose, input, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Transpose, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_permutation(&self) -> Permutation {
        let attr_name = StringBacked::from(Permutation::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Permutation::from(*attr.get())
    }
}

impl Vecmat {
    fn new_maps(op: &mut Self) -> Self {
        let context = op.as_operation().get_context();
        let indexing_maps = Self::get_default_indexing_maps(&context).as_named_attribute();
        op.as_operation().set_attribute_inherent(
            &indexing_maps.get_identifier().as_string(),
            &indexing_maps.as_attribute(),
        );
        op.clone()
    }

    pub fn new_memref(
        context: &Context,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::new_maps(&mut Self::__new_memref(
            &Op::Vecmat,
            context,
            lhs,
            rhs,
            output,
            loc,
        ))
    }

    pub fn new_tensor(
        t: &RankedTensor,
        lhs: &Value,
        rhs: &Value,
        output: &Value,
        loc: &Location,
    ) -> Self {
        Self::new_maps(&mut Self::__new_tensor(
            &Op::Vecmat,
            t,
            lhs,
            rhs,
            output,
            loc,
        ))
    }

    fn check_operands(lhs: &Value, rhs: &Value, output: &Value) -> () {
        Self::__check_operands(&Op::Vecmat, lhs, rhs, output)
    }

    fn check_result(t: &RankedTensor, t_out: &Type) -> () {
        Self::__check_result(&Op::Vecmat, t, t_out)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_default_indexing_maps(context: &Context) -> IndexingMaps {
        let affine_d0 = AffineDim::new(context, 0).as_expr();
        let affine_d1 = AffineDim::new(context, 1).as_expr();
        let map0 = AffineMap::new_results(context, 2, 0, &[affine_d1]);
        let map1 = AffineMap::new_results(context, 2, 0, &[affine_d0, affine_d1]);
        let map2 = AffineMap::new_results(context, 2, 0, &[affine_d0]);
        IndexingMaps::new(context, &[map0, map1, map2])
    }

    pub fn get_indexing_maps(&self) -> IndexingMaps {
        let attr_name = StringBacked::from(IndexingMaps::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        IndexingMaps::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Yield {
    pub fn new(context: &Context, values: &[Value], loc: &Location) -> Self {
        let dialect = context.get_dialect_linalg();
        let name = dialect.get_op_name(&Op::Yield);
        let mut region = Region::new();
        region.append_block(&mut Block::new_empty());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(values);
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

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Abs);

impl ElementwiseUnaryOperationGetBody for Abs {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Abs {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Abs {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Abs.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Abs
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Add);

impl ElementwiseBinaryOperationGetBody for Add {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op = arith::AddF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            vec![op]
        } else if t.is_integer() {
            let op = arith::AddI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            vec![op]
        } else {
            eprintln!("Expected float or integer element type for add operation");
            exit(ExitCode::DialectError);
        }
    }
}

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
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
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
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<i32> for BinaryFunctionKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

SpecializedAttribute!("linalg.fun" = impl NamedInteger for BinaryFunction {});

impl From<i32> for CastKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

SpecializedAttribute!("cast" = impl NamedParsed for Cast {});

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Ceil);

impl ElementwiseUnaryOperationGetBody for Ceil {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Ceil {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Ceil {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Ceil.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Ceil
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpPromotableTypeOperandsAndShapedResult!(Copy);

impl ElementwiseUnaryOperationGetBody for Copy {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Copy {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Copy {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Copy.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Copy
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Div);

impl ElementwiseBinaryOperationGetBody for Div {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op = arith::DivF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            vec![op]
        } else if t.is_integer() {
            let op = arith::DivSI::new(t, lhs, rhs, loc).as_operation();
            vec![op]
        } else {
            eprintln!("Expected float or integer element type for div operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for Div {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Div {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Div.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Div
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(DivUnsigned);

impl ElementwiseBinaryOperationGetBody for DivUnsigned {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_integer() {
            let op = arith::DivUI::new(t, lhs, rhs, loc).as_operation();
            vec![op]
        } else {
            eprintln!("Expected signless integer element type for div unsigned operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for DivUnsigned {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DivUnsigned {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DivUnsigned.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::DivUnsigned
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpPromotableTypeOperandsAndScalarResult!(Dot);

impl ElementwiseBinaryOperationGetBody for Dot {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Dot {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Dot {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Dot.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Dot
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpPromotableTypeOperandsAndShapedResult!(ElementwiseBinary);

impl ElementwiseBinaryOperationGetBody for ElementwiseBinary {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for ElementwiseBinary {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ElementwiseBinary {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ElementwiseBinary.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::ElementwiseBinary
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpPromotableTypeOperandsAndShapedResult!(ElementwiseUnary);

impl ElementwiseUnaryOperationGetBody for ElementwiseUnary {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for ElementwiseUnary {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ElementwiseUnary {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ElementwiseUnary.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::ElementwiseUnary
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Erf);

impl ElementwiseUnaryOperationGetBody for Erf {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Erf {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Erf {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Erf.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Erf
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Exp);

impl ElementwiseUnaryOperationGetBody for Exp {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Exp {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Exp {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Exp.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Exp
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Floor);

impl ElementwiseUnaryOperationGetBody for Floor {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Floor {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Floor {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Floor.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Floor
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<MlirOperation> for Index {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Index {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Index.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Index
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

SpecializedAttribute!("linalg.memoized_indexing_maps" = impl NamedArrayOfAffineMaps for IndexingMaps {});

impl From<i32> for IteratorTypeKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

SpecializedAttribute!("iterator_type" = impl NamedInteger for IteratorType {});

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Log);

impl ElementwiseUnaryOperationGetBody for Log {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Log {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Log {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Log.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Log
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatmulTypeOperandsAndShapedResult!(Matmul);

impl ElementwiseBinaryOperationGetBody for Matmul {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op_mul =
                arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            let op_add = arith::AddF::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::FastMathFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else if t.is_integer() {
            let op_mul = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            let op_add = arith::AddI::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::IntegerOverflowFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else {
            eprintln!("Expected float or integer element type for matmul operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for Matmul {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Matmul {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Matmul.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Matmul
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatmulTransposeATypeOperandsAndShapedResult!(MatmulTransposeA);

impl ElementwiseBinaryOperationGetBody for MatmulTransposeA {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op_mul =
                arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            let op_add = arith::AddF::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::FastMathFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else if t.is_integer() {
            let op_mul = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            let op_add = arith::AddI::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::IntegerOverflowFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else {
            eprintln!("Expected float or integer element type for matmul operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for MatmulTransposeA {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MatmulTransposeA {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MatmulTransposeA.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::MatmulTransposeA
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatmulTransposeBTypeOperandsAndShapedResult!(MatmulTransposeB);

impl ElementwiseBinaryOperationGetBody for MatmulTransposeB {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op_mul =
                arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            let op_add = arith::AddF::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::FastMathFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else if t.is_integer() {
            let op_mul = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            let op_add = arith::AddI::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::IntegerOverflowFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else {
            eprintln!("Expected float or integer element type for matmul operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for MatmulTransposeB {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for MatmulTransposeB {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::MatmulTransposeB.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::MatmulTransposeB
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatvecTypeOperandsAndShapedResult!(Matvec);

impl ElementwiseBinaryOperationGetBody for Matvec {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op_mul =
                arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            let op_add = arith::AddF::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::FastMathFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else if t.is_integer() {
            let op_mul = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            let op_add = arith::AddI::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::IntegerOverflowFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else {
            eprintln!("Expected float or integer element type for matvec operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for Matvec {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Matvec {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Matvec.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Matvec
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Max);

impl ElementwiseBinaryOperationGetBody for Max {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Max {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Max {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Max.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Max
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Min);

impl ElementwiseBinaryOperationGetBody for Min {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Min {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Min {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Min.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Min
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Mul);

impl ElementwiseBinaryOperationGetBody for Mul {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op = arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            vec![op]
        } else if t.is_integer() {
            let op = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            vec![op]
        } else {
            eprintln!("Expected float or integer element type for mul operation");
            exit(ExitCode::DialectError);
        }
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
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Mul.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Mul
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(NegF);

impl ElementwiseUnaryOperationGetBody for NegF {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
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
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
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
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<BinaryFunctionKind> for Op {
    /// NOTE: There is no unsigned equivalent `BinaryFunctionKind` for `Max` and `Min`
    fn from(k: BinaryFunctionKind) -> Self {
        match k {
            BinaryFunctionKind::Add => Op::Add,
            BinaryFunctionKind::Sub => Op::Sub,
            BinaryFunctionKind::Mul => Op::Mul,
            BinaryFunctionKind::Div => Op::Div,
            BinaryFunctionKind::DivUnsigned => Op::DivUnsigned,
            BinaryFunctionKind::MaxSigned => Op::Max,
            BinaryFunctionKind::MinSigned => Op::Min,
            BinaryFunctionKind::MaxUnsigned => Op::Max,
            BinaryFunctionKind::MinUnsigned => Op::Min,
            BinaryFunctionKind::PowF => Op::PowF,
        }
    }
}

impl From<UnaryFunctionKind> for Op {
    fn from(k: UnaryFunctionKind) -> Self {
        match k {
            UnaryFunctionKind::Exp => Op::Exp,
            UnaryFunctionKind::Log => Op::Log,
            UnaryFunctionKind::Abs => Op::Abs,
            UnaryFunctionKind::Ceil => Op::Ceil,
            UnaryFunctionKind::Floor => Op::Floor,
            UnaryFunctionKind::NegF => Op::NegF,
            UnaryFunctionKind::Reciprocal => Op::Reciprocal,
            UnaryFunctionKind::Round => Op::Round,
            UnaryFunctionKind::Sqrt => Op::Sqrt,
            UnaryFunctionKind::Rsqrt => Op::Rsqrt,
            UnaryFunctionKind::Square => Op::Sqrt,
            UnaryFunctionKind::Tanh => Op::Tanh,
            UnaryFunctionKind::Erf => Op::Erf,
        }
    }
}

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

SpecializedAttribute!("permutation" = impl NamedI64DenseArray for Permutation {});

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Reciprocal);

impl ElementwiseUnaryOperationGetBody for Reciprocal {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Reciprocal {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Reciprocal {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Reciprocal.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Reciprocal
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Round);

impl ElementwiseUnaryOperationGetBody for Round {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Round {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Round {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Round.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Round
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Rsqrt);

impl ElementwiseUnaryOperationGetBody for Rsqrt {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Rsqrt {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Rsqrt {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Rsqrt.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Rsqrt
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Sqrt);

impl ElementwiseUnaryOperationGetBody for Sqrt {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Sqrt {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Sqrt {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Sqrt.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Sqrt
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Square);

impl ElementwiseUnaryOperationGetBody for Square {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Square {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Square {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Square.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Square
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseBinaryOpMatchingTypeOperandsAndShapedResult!(Sub);

impl ElementwiseBinaryOperationGetBody for Sub {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op = arith::SubF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            vec![op]
        } else if t.is_integer() {
            let op = arith::SubI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            vec![op]
        } else {
            eprintln!("Expected float or integer element type for sub operation");
            exit(ExitCode::DialectError);
        }
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
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Sub.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Sub
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndShapedResult!(Tanh);

impl ElementwiseUnaryOperationGetBody for Tanh {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        todo!()
    }
}

impl From<MlirOperation> for Tanh {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Tanh {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Tanh.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Tanh
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl_ElementwiseUnaryOpMatchingTypeOperandsAndTransposeResult!(Transpose);

impl ElementwiseUnaryOperationGetBody for Transpose {
    fn get_body(t: &Type, input: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() || t.is_integer() {
            vec![]
        } else {
            eprintln!("Expected float or integer element type for matmul operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for Transpose {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Transpose {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::OpAsmOpInterface,
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Transpose.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Transpose
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<i32> for UnaryFunctionKind {
    fn from(n: i32) -> Self {
        Self::from_i32(n)
    }
}

SpecializedAttribute!("linalg.fun" = impl NamedInteger for UnaryFunction {});

impl_ElementwiseBinaryOpVecmatTypeOperandsAndShapedResult!(Vecmat);

impl ElementwiseBinaryOperationGetBody for Vecmat {
    fn get_body(t: &Type, lhs: &Value, rhs: &Value, acc: &Value, loc: &Location) -> Vec<Operation> {
        if t.is_float() {
            let op_mul =
                arith::MulF::new(t, lhs, rhs, arith::FastMathFlags::None, loc).as_operation();
            let op_add = arith::AddF::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::FastMathFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else if t.is_integer() {
            let op_mul = arith::MulI::new(t, lhs, rhs, arith::IntegerOverflowFlags::None, loc)
                .as_operation();
            let op_add = arith::AddI::new(
                t,
                acc,
                &op_mul.get_result(0),
                arith::IntegerOverflowFlags::None,
                loc,
            )
            .as_operation();
            vec![op_mul, op_add]
        } else {
            eprintln!("Expected float or integer element type for vecmat operation");
            exit(ExitCode::DialectError);
        }
    }
}

impl From<MlirOperation> for Vecmat {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Vecmat {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::DestinationStyleOpInterface,
            Interface::LinalgContractionOpInterface,
            Interface::LinalgStructuredInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::RecursiveMemoryEffects),
            Interface::ReifyRankedShapeTypeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Vecmat.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Vecmat
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlock,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
        ]
    }
}

impl From<MlirOperation> for Yield {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Yield {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_linalg()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Yield
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

impl<T: Shape> TransformShape for T {
    fn matmul(&self, rhs: &Self) -> Option<Vec<i64>> {
        if self.rank() == 2 && rhs.rank() == 2 && self.get(1) == rhs.get(0) {
            Some(vec![self.get(0), rhs.get(1)])
        } else {
            None
        }
    }

    fn matmul_transpose_a(&self, rhs: &Self) -> Option<Vec<i64>> {
        if self.rank() == 2 && rhs.rank() == 2 && self.get(0) == rhs.get(0) {
            Some(vec![self.get(1), rhs.get(1)])
        } else {
            None
        }
    }

    fn matmul_transpose_b(&self, rhs: &Self) -> Option<Vec<i64>> {
        if self.rank() == 2 && rhs.rank() == 2 && self.get(1) == rhs.get(1) {
            Some(vec![self.get(0), rhs.get(0)])
        } else {
            None
        }
    }

    fn matvec(&self, rhs: &Self) -> Option<Vec<i64>> {
        if self.rank() == 2 && rhs.rank() == 1 && self.get(1) == rhs.get(0) {
            Some(vec![self.get(0)])
        } else {
            None
        }
    }

    fn vecmat(&self, rhs: &Self) -> Option<Vec<i64>> {
        if self.rank() == 1 && rhs.rank() == 2 && self.get(0) == rhs.get(0) {
            Some(vec![rhs.get(1)])
        } else {
            None
        }
    }

    fn unpack_matmul(&self, rhs: &Self) -> Option<(isize, Vec<i64>)> {
        self.matmul(rhs).map(|v| (self.rank(), v))
    }

    fn unpack_matmul_transpose_a(&self, rhs: &Self) -> Option<(isize, Vec<i64>)> {
        self.matmul_transpose_a(rhs).map(|v| (self.rank(), v))
    }

    fn unpack_matmul_transpose_b(&self, rhs: &Self) -> Option<(isize, Vec<i64>)> {
        self.matmul_transpose_b(rhs).map(|v| (self.rank(), v))
    }

    fn unpack_matvec(&self, rhs: &Self) -> Option<(isize, Vec<i64>)> {
        self.matvec(rhs).map(|v| (rhs.rank(), v))
    }

    fn unpack_vecmat(&self, rhs: &Self) -> Option<(isize, Vec<i64>)> {
        self.vecmat(rhs).map(|v| (self.rank(), v))
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for BinaryFunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            BinaryFunctionKind::Add => "Add",
            BinaryFunctionKind::Sub => "Sub",
            BinaryFunctionKind::Mul => "Mul",
            BinaryFunctionKind::Div => "Div",
            BinaryFunctionKind::DivUnsigned => "DivUnsigned",
            BinaryFunctionKind::MaxSigned => "MaxSigned",
            BinaryFunctionKind::MinSigned => "MinSigned",
            BinaryFunctionKind::MaxUnsigned => "MaxUnsigned",
            BinaryFunctionKind::MinUnsigned => "MinUnsigned",
            BinaryFunctionKind::PowF => "PowF",
        })
    }
}

impl fmt::Display for CastKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            CastKind::Signed => "Signed",
            CastKind::Unsigned => "Unsigned",
        })
    }
}

impl fmt::Display for IteratorTypeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            IteratorTypeKind::Parallel => "Parallel",
            IteratorTypeKind::Reduction => "Reduction",
            IteratorTypeKind::Window => "Window",
        })
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Abs => "AbsOp",
            Op::Add => "AddOp",
            Op::BatchMatmul => "BatchMatmulOp",
            Op::BatchMatmulTransposeA => "BatchMatmulTransposeAOp",
            Op::BatchMatmulTransposeB => "BatchMatmulTransposeBOp",
            Op::BatchMatvec => "BatchMatvecOp",
            Op::BatchMmt4D => "BatchMmt4DOp",
            Op::BatchReduceMatmul => "BatchReduceMatmulOp",
            Op::BatchVecmat => "BatchVecmatOp",
            Op::Broadcast => "BroadcastOp",
            Op::Ceil => "CeilOp",
            Op::Conv1DNcwFcw => "Conv1DNcwFcwOp",
            Op::Conv1DNwcWcf => "Conv1DNwcWcfOp",
            Op::Conv1D => "Conv1DOp",
            Op::Conv2D => "Conv2DOp",
            Op::Conv2DNchwFchw => "Conv2DNchwFchwOp",
            Op::Conv2DNgchwGfchw => "Conv2DNgchwGfchwOp",
            Op::Conv2DNgchwGfchwG => "Conv2DNgchwGfchwGOp",
            Op::Conv2DNhwcFhwc => "Conv2DNhwcFhwcOp",
            Op::Conv2DNhwcFhwcQ => "Conv2DNhwcFhwcQOp",
            Op::Conv2DNhwcHwcf => "Conv2DNhwcHwcfOp",
            Op::Conv2DNhwcHwcfQ => "Conv2DNhwcHwcfQOp",
            Op::Conv3D => "Conv3DOp",
            Op::Conv3DNcdhwFcdhw => "Conv3DNcdhwFcdhwOp",
            Op::Conv3DNdhwcDhwcf => "Conv3DNdhwcDhwcfOp",
            Op::Conv3DNdhwcDhwcfQ => "Conv3DNdhwcDhwcfQOp",
            Op::Copy => "CopyOp",
            Op::DepthwiseConv1DNcwCw => "DepthwiseConv1DNcwCwOp",
            Op::DepthwiseConv1DNwcWc => "DepthwiseConv1DNwcWcOp",
            Op::DepthwiseConv1DNwcWcm => "DepthwiseConv1DNwcWcmOp",
            Op::DepthwiseConv2DNchwChw => "DepthwiseConv2DNchwChwOp",
            Op::DepthwiseConv2DNhwcHwc => "DepthwiseConv2DNhwcHwcOp",
            Op::DepthwiseConv2DNhwcHwcQ => "DepthwiseConv2DNhwcHwcQOp",
            Op::DepthwiseConv2DNhwcHwcm => "DepthwiseConv2DNhwcHwcmOp",
            Op::DepthwiseConv2DNhwcHwcmQ => "DepthwiseConv2DNhwcHwcmQOp",
            Op::DepthwiseConv3DNcdhwCdhw => "DepthwiseConv3DNcdhwCdhwOp",
            Op::DepthwiseConv3DNdhwcDhwc => "DepthwiseConv3DNdhwcDhwcOp",
            Op::DepthwiseConv3DNdhwcDhwcm => "DepthwiseConv3DNdhwcDhwcmOp",
            Op::Div => "DivOp",
            Op::DivUnsigned => "DivUnsignedOp",
            Op::Dot => "DotOp",
            Op::ElementwiseBinary => "ElementwiseBinaryOp",
            Op::ElementwiseUnary => "ElementwiseUnaryOp",
            Op::Erf => "ErfOp",
            Op::Exp => "ExpOp",
            Op::Fill => "FillOp",
            Op::FillRng2D => "FillRng2DOp",
            Op::Floor => "FloorOp",
            Op::Generic => "GenericOp",
            Op::Index => "IndexOp",
            Op::Log => "LogOp",
            Op::Map => "MapOp",
            Op::Matmul => "MatmulOp",
            Op::MatmulTransposeA => "MatmulTransposeAOp",
            Op::MatmulTransposeB => "MatmulTransposeBOp",
            Op::Matvec => "MatvecOp",
            Op::Max => "MaxOp",
            Op::Min => "MinOp",
            Op::Mmt4D => "Mmt4DOp",
            Op::Mul => "MulOp",
            Op::NegF => "NegFOp",
            Op::PoolingNchwMax => "PoolingNchwMaxOp",
            Op::PoolingNchwSum => "PoolingNchwSumOp",
            Op::PoolingNcwMax => "PoolingNcwMaxOp",
            Op::PoolingNcwSum => "PoolingNcwSumOp",
            Op::PoolingNdhwcMax => "PoolingNdhwcMaxOp",
            Op::PoolingNdhwcMin => "PoolingNdhwcMinOp",
            Op::PoolingNdhwcSum => "PoolingNdhwcSumOp",
            Op::PoolingNhwcMax => "PoolingNhwcMaxOp",
            Op::PoolingNhwcMaxUnsigned => "PoolingNhwcMaxUnsignedOp",
            Op::PoolingNhwcMin => "PoolingNhwcMinOp",
            Op::PoolingNhwcMinUnsigned => "PoolingNhwcMinUnsignedOp",
            Op::PoolingNhwcSum => "PoolingNhwcSumOp",
            Op::PoolingNwcMax => "PoolingNwcMaxOp",
            Op::PoolingNwcMaxUnsigned => "PoolingNwcMaxUnsignedOp",
            Op::PoolingNwcMin => "PoolingNwcMinOp",
            Op::PoolingNwcMinUnsigned => "PoolingNwcMinUnsignedOp",
            Op::PoolingNwcSum => "PoolingNwcSumOp",
            Op::PowF => "PowFOp",
            Op::QuantizedBatchMatmul => "QuantizedBatchMatmulOp",
            Op::QuantizedMatmul => "QuantizedMatmulOp",
            Op::Reciprocal => "ReciprocalOp",
            Op::Reduce => "ReduceOp",
            Op::Round => "RoundOp",
            Op::Rsqrt => "RsqrtOp",
            Op::Select => "SelectOp",
            Op::Softmax => "SoftmaxOp",
            Op::Sqrt => "SqrtOp",
            Op::Square => "SquareOp",
            Op::Sub => "SubOp",
            Op::Tanh => "TanhOp",
            Op::Transpose => "TransposeOp",
            Op::Vecmat => "VecmatOp",
            Op::WinogradFilterTransform => "WinogradFilterTransformOp",
            Op::WinogradInputTransform => "WinogradInputTransformOp",
            Op::WinogradOutputTransform => "WinogradOutputTransformOp",
            Op::Yield => "YieldOp",
        })
    }
}

impl fmt::Display for UnaryFunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            UnaryFunctionKind::Exp => "Exp",
            UnaryFunctionKind::Log => "Log",
            UnaryFunctionKind::Abs => "Abs",
            UnaryFunctionKind::Ceil => "Ceil",
            UnaryFunctionKind::Floor => "Floor",
            UnaryFunctionKind::NegF => "NegF",
            UnaryFunctionKind::Reciprocal => "Reciprocal",
            UnaryFunctionKind::Round => "Round",
            UnaryFunctionKind::Sqrt => "Sqrt",
            UnaryFunctionKind::Rsqrt => "Rsqrt",
            UnaryFunctionKind::Square => "Square",
            UnaryFunctionKind::Tanh => "Tanh",
            UnaryFunctionKind::Erf => "Erf",
        })
    }
}
