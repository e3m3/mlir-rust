// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAffineExpr;
use mlir_sys::MlirAffineMap;
use mlir_sys::MlirAttribute;
use mlir_sys::MlirIntegerSet;
use mlir_sys::MlirOperation;
use mlir_sys::mlirAffineAddExprGet;
use mlir_sys::mlirAffineBinaryOpExprGetLHS;
use mlir_sys::mlirAffineBinaryOpExprGetRHS;
use mlir_sys::mlirAffineCeilDivExprGet;
use mlir_sys::mlirAffineConstantExprGet;
use mlir_sys::mlirAffineConstantExprGetValue;
use mlir_sys::mlirAffineDimExprGet;
use mlir_sys::mlirAffineDimExprGetPosition;
use mlir_sys::mlirAffineExprCompose;
use mlir_sys::mlirAffineExprDump;
use mlir_sys::mlirAffineExprEqual;
use mlir_sys::mlirAffineExprGetContext;
use mlir_sys::mlirAffineExprGetLargestKnownDivisor;
use mlir_sys::mlirAffineExprIsAAdd;
use mlir_sys::mlirAffineExprIsABinary;
use mlir_sys::mlirAffineExprIsACeilDiv;
use mlir_sys::mlirAffineExprIsAConstant;
use mlir_sys::mlirAffineExprIsADim;
use mlir_sys::mlirAffineExprIsAFloorDiv;
use mlir_sys::mlirAffineExprIsAMod;
use mlir_sys::mlirAffineExprIsAMul;
use mlir_sys::mlirAffineExprIsASymbol;
use mlir_sys::mlirAffineExprIsFunctionOfDim;
use mlir_sys::mlirAffineExprIsMultipleOf;
use mlir_sys::mlirAffineExprIsPureAffine;
use mlir_sys::mlirAffineExprIsSymbolicOrConstant;
use mlir_sys::mlirAffineExprPrint;
use mlir_sys::mlirAffineFloorDivExprGet;
use mlir_sys::mlirAffineMapAttrGet;
use mlir_sys::mlirAffineMapAttrGetTypeID;
use mlir_sys::mlirAffineMapAttrGetValue;
use mlir_sys::mlirAffineMapConstantGet;
use mlir_sys::mlirAffineMapDump;
use mlir_sys::mlirAffineMapEmptyGet;
use mlir_sys::mlirAffineMapEqual;
use mlir_sys::mlirAffineMapGet;
use mlir_sys::mlirAffineMapGetContext;
use mlir_sys::mlirAffineMapGetMajorSubMap;
use mlir_sys::mlirAffineMapGetMinorSubMap;
use mlir_sys::mlirAffineMapGetNumDims;
use mlir_sys::mlirAffineMapGetNumInputs;
use mlir_sys::mlirAffineMapGetNumResults;
use mlir_sys::mlirAffineMapGetNumSymbols;
use mlir_sys::mlirAffineMapGetResult;
use mlir_sys::mlirAffineMapGetSingleConstantResult;
use mlir_sys::mlirAffineMapGetSubMap;
use mlir_sys::mlirAffineMapIsEmpty;
use mlir_sys::mlirAffineMapIsIdentity;
use mlir_sys::mlirAffineMapIsMinorIdentity;
use mlir_sys::mlirAffineMapIsPermutation;
use mlir_sys::mlirAffineMapIsProjectedPermutation;
use mlir_sys::mlirAffineMapIsSingleConstant;
use mlir_sys::mlirAffineMapMinorIdentityGet;
use mlir_sys::mlirAffineMapMultiDimIdentityGet;
use mlir_sys::mlirAffineMapPermutationGet;
use mlir_sys::mlirAffineMapPrint;
use mlir_sys::mlirAffineMapReplace;
use mlir_sys::mlirAffineMapZeroResultGet;
use mlir_sys::mlirAffineModExprGet;
use mlir_sys::mlirAffineMulExprGet;
use mlir_sys::mlirAffineSymbolExprGet;
use mlir_sys::mlirAffineSymbolExprGetPosition;
use mlir_sys::mlirIntegerSetDump;
use mlir_sys::mlirIntegerSetEmptyGet;
use mlir_sys::mlirIntegerSetEqual;
use mlir_sys::mlirIntegerSetGet;
use mlir_sys::mlirIntegerSetGetConstraint;
use mlir_sys::mlirIntegerSetGetContext;
use mlir_sys::mlirIntegerSetGetNumConstraints;
use mlir_sys::mlirIntegerSetGetNumDims;
use mlir_sys::mlirIntegerSetGetNumEqualities;
use mlir_sys::mlirIntegerSetGetNumInequalities;
use mlir_sys::mlirIntegerSetGetNumInputs;
use mlir_sys::mlirIntegerSetGetNumSymbols;
use mlir_sys::mlirIntegerSetIsCanonicalEmpty;
use mlir_sys::mlirIntegerSetIsConstraintEq;
use mlir_sys::mlirIntegerSetPrint;
use mlir_sys::mlirIntegerSetReplaceGet;

use std::cmp;
use std::ffi::c_uint;
use std::fmt;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::IAttributeNamed;
use attributes::specialized::NamedAffineMap;
use attributes::specialized::NamedAffineSet;
use attributes::specialized::NamedArrayOfIntegers;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI32DenseElements;
use attributes::specialized::NamedIndex;
use attributes::specialized::SpecializedAttribute;
use dialects::IOp;
use dialects::IOperation;
use dialects::OpRef;
use dialects::arith::AtomicRMWKind;
use dialects::common::OperandSegmentSizes;
use effects::MEFF_NO_MEMORY_EFFECT;
use effects::MemoryEffectList;
use exit_code::ExitCode;
use exit_code::exit;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Attribute;
use ir::Block;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::Operation;
use ir::OperationState;
use ir::Region;
use ir::ShapeImpl;
use ir::StringBacked;
use ir::StringCallback;
use ir::StringCallbackState;
use ir::Type;
use ir::TypeID;
use ir::Value;
use ir::print_method;
use traits::Trait;
use types::IType;
use types::index::Index;
use types::integer::Integer as IntegerType;
use types::ranked_tensor::RankedTensor;
use types::shaped::Shaped;
use types::vector::Vector;

///////////////////////////////
//  Traits
///////////////////////////////

pub trait IExpr {
    fn as_expr(&self) -> Expr;
    fn get(&self) -> &MlirAffineExpr;
}

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct Condition(MlirAttribute);

#[derive(Clone)]
pub struct LowerBoundsFor(MlirAttribute);

#[derive(Clone)]
pub struct LowerBoundGroups(MlirAttribute);

#[derive(Clone)]
pub struct LowerBoundsParallel(MlirAttribute);

#[derive(Clone)]
pub struct NamedMap(MlirAttribute);

#[derive(Clone)]
pub struct NamedMapSource(MlirAttribute);

#[derive(Clone)]
pub struct NamedMapTag(MlirAttribute);

#[derive(Clone)]
pub struct NamedMapTarget(MlirAttribute);

#[derive(Clone)]
pub struct Reductions(MlirAttribute);

#[derive(Clone)]
pub struct StepFor(MlirAttribute);

#[derive(Clone)]
pub struct StepsParallel(MlirAttribute);

#[derive(Clone)]
pub struct UpperBoundsFor(MlirAttribute);

#[derive(Clone)]
pub struct UpperBoundGroups(MlirAttribute);

#[derive(Clone)]
pub struct UpperBoundsParallel(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[derive(Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Mod,
    Mul,
    CeilDiv,
    FloorDiv,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Apply,
    DelinearizeIndex,
    DmaStart,
    DmaWait,
    For,
    If,
    LinearizeIndex,
    Load,
    Max,
    Min,
    Parallel,
    Prefetch,
    Store,
    VectorLoad,
    VectorStore,
    Yield,
}

pub type ReductionOp = AtomicRMWKind;

///////////////////////////////
//  Exprs
///////////////////////////////

#[derive(Clone, Copy)]
pub struct Binary(MlirAffineExpr, BinOp);

#[derive(Clone, Copy)]
pub struct Constant(MlirAffineExpr);

#[derive(Clone, Copy)]
pub struct Dim(MlirAffineExpr);

#[derive(Clone, Copy)]
pub struct Expr(MlirAffineExpr);

#[derive(Clone, Copy)]
pub struct Map(MlirAffineMap);

#[derive(Clone, Copy)]
pub struct Set(MlirIntegerSet);

#[derive(Clone, Copy)]
pub struct Symbol(MlirAffineExpr);

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Apply(MlirOperation);

#[derive(Clone)]
pub struct DelinearizeIndex(MlirOperation);

#[derive(Clone)]
pub struct DmaStart(MlirOperation);

#[derive(Clone)]
pub struct DmaWait(MlirOperation);

#[derive(Clone)]
pub struct For(MlirOperation);

#[derive(Clone)]
pub struct If(MlirOperation);

#[derive(Clone)]
pub struct LinearizeIndex(MlirOperation);

#[derive(Clone)]
pub struct Load(MlirOperation);

#[derive(Clone)]
pub struct Max(MlirOperation);

#[derive(Clone)]
pub struct Min(MlirOperation);

#[derive(Clone)]
pub struct Parallel(MlirOperation);

#[derive(Clone)]
pub struct Prefetch(MlirOperation);

#[derive(Clone)]
pub struct Store(MlirOperation);

#[derive(Clone)]
pub struct VectorLoad(MlirOperation);

#[derive(Clone)]
pub struct VectorStore(MlirOperation);

#[derive(Clone)]
pub struct Yield(MlirOperation, MlirOperation, Op);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl Condition {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl LowerBoundsFor {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl LowerBoundGroups {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl LowerBoundsParallel {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl NamedMap {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl NamedMapSource {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl NamedMapTag {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl NamedMapTarget {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Reductions {
    pub fn new(context: &Context, ops: &[ReductionOp]) -> Self {
        let r: Vec<i64> = ops.iter().map(|&op| op as i64).collect();
        <Self as NamedArrayOfIntegers>::new_i64(context, &r)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StepFor {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StepsParallel {
    pub fn new(context: &Context, ops: &[i64]) -> Self {
        <Self as NamedArrayOfIntegers>::new_i64(context, ops)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl UpperBoundsFor {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl UpperBoundGroups {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl UpperBoundsParallel {
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
            Op::Apply => "apply",
            Op::DelinearizeIndex => "delinearize_index",
            Op::DmaStart => "dma_start",
            Op::DmaWait => "dma_wait",
            Op::For => "for",
            Op::If => "if",
            Op::LinearizeIndex => "linearize_index",
            Op::Load => "load",
            Op::Max => "max",
            Op::Min => "min",
            Op::Parallel => "parallel",
            Op::Prefetch => "prefetch",
            Op::Store => "store",
            Op::VectorLoad => "vector_load",
            Op::VectorStore => "vector_store",
            Op::Yield => "yield",
        }
    }
}

///////////////////////////////
//  Expr Implementation
///////////////////////////////

impl Binary {
    pub fn new_add(lhs: Expr, rhs: Expr) -> Self {
        Self::from((
            do_unsafe!(mlirAffineAddExprGet(*lhs.get(), *rhs.get())),
            BinOp::Add,
        ))
    }

    pub fn new_mod(lhs: Expr, rhs: Expr) -> Self {
        Self::from((
            do_unsafe!(mlirAffineModExprGet(*lhs.get(), *rhs.get())),
            BinOp::Mod,
        ))
    }

    pub fn new_mul(lhs: Expr, rhs: Expr) -> Self {
        Self::from((
            do_unsafe!(mlirAffineMulExprGet(*lhs.get(), *rhs.get())),
            BinOp::Mul,
        ))
    }

    pub fn new_ceil_div(lhs: Expr, rhs: Expr) -> Self {
        Self::from((
            do_unsafe!(mlirAffineCeilDivExprGet(*lhs.get(), *rhs.get())),
            BinOp::CeilDiv,
        ))
    }

    pub fn new_floor_div(lhs: Expr, rhs: Expr) -> Self {
        Self::from((
            do_unsafe!(mlirAffineFloorDivExprGet(*lhs.get(), *rhs.get())),
            BinOp::FloorDiv,
        ))
    }

    pub fn new_sub(lhs: Expr, rhs_: Expr) -> Self {
        let context = lhs.get_context();
        let cn1 = Constant::new(&context, -1).as_expr();
        let rhs = Self::new_mul(rhs_, cn1).as_expr();
        Self::from((
            do_unsafe!(mlirAffineAddExprGet(*lhs.get(), *rhs.get())),
            BinOp::Add,
        ))
    }

    pub fn from_checked(expr: MlirAffineExpr, op: BinOp) -> Self {
        let expr_ = Expr::from(expr);
        // NOTE: Incoming expression may have been internally simplified to a constant (e.g., const * -1).
        if !expr_.is_binary() && !expr_.is_constant() {
            eprint!("Affine expression given cannot be coerced to a binary operation: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self::from((expr, op))
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_lhs(&self) -> Expr {
        Expr::from(do_unsafe!(mlirAffineBinaryOpExprGetLHS(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineExpr {
        &mut self.0
    }

    pub fn get_op(&self) -> BinOp {
        self.1
    }

    pub fn get_rhs(&self) -> Expr {
        Expr::from(do_unsafe!(mlirAffineBinaryOpExprGetRHS(*self.get())))
    }
}

impl Constant {
    pub fn new(context: &Context, constant: i64) -> Self {
        Self::from(do_unsafe!(mlirAffineConstantExprGet(
            *context.get(),
            constant
        )))
    }

    pub fn from_checked(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_constant() {
            eprint!("Affine expression given cannot be coerced to a constant: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self::from(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineExpr {
        &mut self.0
    }

    pub fn get_value(&self) -> i64 {
        do_unsafe!(mlirAffineConstantExprGetValue(*self.get()))
    }
}

impl Dim {
    pub fn new(context: &Context, i: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineDimExprGet(*context.get(), i)))
    }

    pub fn from_checked(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_dim() {
            eprint!("Affine expression given cannot be coerced to a dimension: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self::from(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_position(&self) -> isize {
        do_unsafe!(mlirAffineDimExprGetPosition(*self.get()))
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineExpr {
        &mut self.0
    }
}

impl Expr {
    pub fn as_binary(&self) -> Option<Binary> {
        if self.is_binary() {
            let op = if self.is_add() {
                BinOp::Add
            } else if self.is_mod() {
                BinOp::Mod
            } else if self.is_mul() {
                BinOp::Mul
            } else if self.is_ceil_div() {
                BinOp::CeilDiv
            } else if self.is_floor_div() {
                BinOp::FloorDiv
            } else {
                return None;
            };
            Some(Binary::from((*self.get(), op)))
        } else {
            None
        }
    }

    pub fn as_constant(&self) -> Option<Constant> {
        if self.is_constant() {
            Some(Constant::from(*self.get()))
        } else {
            None
        }
    }

    pub fn as_dim(&self) -> Option<Dim> {
        if self.is_dim() {
            Some(Dim::from(*self.get()))
        } else {
            None
        }
    }

    pub fn as_symbol(&self) -> Option<Symbol> {
        if self.is_symbol() {
            Some(Symbol::from(*self.get()))
        } else {
            None
        }
    }

    pub fn compose(&self, map: &Map) -> Self {
        Self::from(do_unsafe!(mlirAffineExprCompose(*self.get(), *map.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirAffineExprDump(*self.get()))
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirAffineExprGetContext(*self.get())))
    }

    pub fn get_largest_divisor(&self) -> i64 {
        do_unsafe!(mlirAffineExprGetLargestKnownDivisor(*self.get()))
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineExpr {
        &mut self.0
    }

    pub fn is_add(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAAdd(*self.get()))
    }

    pub fn is_mod(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAMod(*self.get()))
    }

    pub fn is_mul(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAMul(*self.get()))
    }

    pub fn is_ceil_div(&self) -> bool {
        do_unsafe!(mlirAffineExprIsACeilDiv(*self.get()))
    }

    pub fn is_floor_div(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAFloorDiv(*self.get()))
    }

    pub fn is_binary(&self) -> bool {
        do_unsafe!(mlirAffineExprIsABinary(*self.get()))
    }

    pub fn is_constant(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAConstant(*self.get()))
    }

    pub fn is_dim(&self) -> bool {
        do_unsafe!(mlirAffineExprIsADim(*self.get()))
    }

    pub fn is_function_of_dim(&self, pos: isize) -> bool {
        do_unsafe!(mlirAffineExprIsFunctionOfDim(*self.get(), pos))
    }

    pub fn is_multiple_of(&self, factor: i64) -> bool {
        do_unsafe!(mlirAffineExprIsMultipleOf(*self.get(), factor))
    }

    pub fn is_pure(&self) -> bool {
        do_unsafe!(mlirAffineExprIsPureAffine(*self.get()))
    }

    pub fn is_semi_affine(&self) -> bool {
        !self.is_pure()
    }

    pub fn is_symbol(&self) -> bool {
        do_unsafe!(mlirAffineExprIsASymbol(*self.get()))
    }

    pub fn is_symbolic_or_constant(&self) -> bool {
        do_unsafe!(mlirAffineExprIsSymbolicOrConstant(*self.get()))
    }

    print_method!(mlirAffineExprPrint);
}

impl Map {
    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirAffineMapEmptyGet(*context.get())))
    }

    pub fn new_constant(context: &Context, constant: i64) -> Self {
        Self::from(do_unsafe!(mlirAffineMapConstantGet(
            *context.get(),
            constant
        )))
    }

    pub fn new_minor_identity(context: &Context, num_dims: isize, num_results: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapMinorIdentityGet(
            *context.get(),
            num_dims,
            num_results
        )))
    }

    pub fn new_identity(context: &Context, num_dims: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapMultiDimIdentityGet(
            *context.get(),
            num_dims
        )))
    }

    pub fn new_permutation(context: &Context, permutation: &mut [c_uint]) -> Self {
        Self::from(do_unsafe!(mlirAffineMapPermutationGet(
            *context.get(),
            permutation.len() as isize,
            permutation.as_mut_ptr(),
        )))
    }

    pub fn new_results(
        context: &Context,
        num_dims: isize,
        num_syms: isize,
        exprs: &[Expr],
    ) -> Self {
        let mut e: Vec<MlirAffineExpr> = exprs.iter().map(|e| *e.get()).collect();
        Self::from(do_unsafe!(mlirAffineMapGet(
            *context.get(),
            num_dims,
            num_syms,
            exprs.len() as isize,
            e.as_mut_ptr(),
        )))
    }

    pub fn new_zero_result(context: &Context, num_dims: isize, num_syms: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapZeroResultGet(
            *context.get(),
            num_dims,
            num_syms
        )))
    }

    pub fn from_attribute(attr: &Attribute) -> Self {
        if !attr.is_affine_map() {
            eprint!("Attribute cannot be coerced to an affine map: ");
            attr.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self::from(do_unsafe!(mlirAffineMapAttrGetValue(*attr.get())))
    }

    pub fn as_attribute(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirAffineMapAttrGet(*self.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirAffineMapDump(*self.get()))
    }

    pub fn get(&self) -> &MlirAffineMap {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirAffineMapGetContext(*self.get())))
    }

    pub fn get_major_sub_map(&self, num_results: isize) -> Option<Self> {
        if num_results <= 0 {
            None
        } else if num_results >= self.num_results() {
            Some(*self)
        } else {
            Some(Self::from(do_unsafe!(mlirAffineMapGetMajorSubMap(
                *self.get(),
                num_results
            ))))
        }
    }

    pub fn get_minor_sub_map(&self, num_results: isize) -> Option<Self> {
        if num_results <= 0 {
            None
        } else if num_results >= self.num_results() {
            Some(*self)
        } else {
            Some(Self::from(do_unsafe!(mlirAffineMapGetMinorSubMap(
                *self.get(),
                num_results
            ))))
        }
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineMap {
        &mut self.0
    }

    pub fn get_result(&self, i: isize) -> Expr {
        if i >= self.num_results() || i < 0 {
            eprint!("Index '{}' out of bounds for affine map results: ", i);
            self.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Expr::from(do_unsafe!(mlirAffineMapGetResult(*self.get(), i)))
    }

    pub fn get_single_constant(&self) -> i64 {
        do_unsafe!(mlirAffineMapGetSingleConstantResult(*self.get()))
    }

    pub fn get_sub_map(&self, pos: &mut [isize]) -> Self {
        Self::from(do_unsafe!(mlirAffineMapGetSubMap(
            *self.get(),
            pos.len() as isize,
            pos.as_mut_ptr()
        )))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirAffineMapAttrGetTypeID()))
    }

    pub fn is_empty(&self) -> bool {
        do_unsafe!(mlirAffineMapIsEmpty(*self.get()))
    }

    pub fn is_identity(&self) -> bool {
        do_unsafe!(mlirAffineMapIsIdentity(*self.get()))
    }

    pub fn is_minor_identity(&self) -> bool {
        do_unsafe!(mlirAffineMapIsMinorIdentity(*self.get()))
    }

    pub fn is_permutation(&self) -> bool {
        do_unsafe!(mlirAffineMapIsPermutation(*self.get()))
    }

    pub fn is_projected_permutation(&self) -> bool {
        do_unsafe!(mlirAffineMapIsProjectedPermutation(*self.get()))
    }

    pub fn is_single_constant(&self) -> bool {
        do_unsafe!(mlirAffineMapIsSingleConstant(*self.get()))
    }

    pub fn num_dims(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumDims(*self.get()))
    }

    /// Equal to `num_dims() + num_symbols()`
    pub fn num_inputs(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumInputs(*self.get()))
    }

    pub fn num_results(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumResults(*self.get()))
    }

    pub fn num_symbols(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumSymbols(*self.get()))
    }

    print_method!(mlirAffineMapPrint);

    pub fn replace_expr(&mut self, old: Expr, new: Expr, num_dims: isize, num_syms: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapReplace(
            *self.get_mut(),
            *old.get(),
            *new.get(),
            num_dims,
            num_syms
        )))
    }
}

impl Set {
    pub fn new(
        context: &Context,
        num_dims: isize,
        num_syms: isize,
        constraints: &[Expr],
        flags: &[bool],
    ) -> Self {
        let c_len = constraints.len();
        let f_len = flags.len();
        if c_len != f_len {
            eprintln!(
                "Mismatched constraints ('{}') and flags ('{}') sizes",
                c_len, f_len
            );
            exit(ExitCode::DialectError);
        }
        let c: Vec<MlirAffineExpr> = constraints.iter().map(|e| *e.get()).collect();
        Self::from(do_unsafe!(mlirIntegerSetGet(
            *context.get(),
            num_dims,
            num_syms,
            c_len as isize,
            c.as_ptr(),
            flags.as_ptr(),
        )))
    }

    pub fn new_empty(context: &Context, num_dims: isize, num_syms: isize) -> Self {
        Self::from(do_unsafe!(mlirIntegerSetEmptyGet(
            *context.get(),
            num_dims,
            num_syms
        )))
    }

    pub fn as_attribute(&self) -> Attribute {
        todo!()
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirIntegerSetDump(*self.get()))
    }

    pub fn get(&self) -> &MlirIntegerSet {
        &self.0
    }

    pub fn get_constraint(&self, pos: usize) -> Expr {
        if pos as isize >= self.num_constraints() {
            eprintln!("Index {} out of bounds for integer set constraints", pos);
            exit(ExitCode::DialectError);
        } else {
            Expr::from(do_unsafe!(mlirIntegerSetGetConstraint(
                *self.get(),
                pos as isize
            )))
        }
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirIntegerSetGetContext(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirIntegerSet {
        &mut self.0
    }

    pub fn is_equality_constraint(&self, pos: usize) -> bool {
        if pos as isize >= self.num_constraints() {
            eprintln!("Index {} out of bounds for integer set constraints", pos);
            exit(ExitCode::DialectError);
        } else {
            do_unsafe!(mlirIntegerSetIsConstraintEq(*self.get(), pos as isize))
        }
    }

    pub fn is_empty_set(&self) -> bool {
        do_unsafe!(mlirIntegerSetIsCanonicalEmpty(*self.get()))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn num_constraints(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumConstraints(*self.get()))
    }

    pub fn num_dims(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumDims(*self.get()))
    }

    pub fn num_equalities(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumEqualities(*self.get()))
    }

    pub fn num_inequalities(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumInequalities(*self.get()))
    }

    pub fn num_inputs(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumInputs(*self.get()))
    }

    pub fn num_symbols(&self) -> isize {
        do_unsafe!(mlirIntegerSetGetNumSymbols(*self.get()))
    }

    print_method!(mlirIntegerSetPrint);

    pub fn replace_dimensions_and_symbols(&mut self, dims: &[Dim], syms: &[Symbol]) -> Self {
        let n_dims = self.num_dims();
        let n_syms = self.num_symbols();
        let n_dims_new = dims.len() as isize;
        let n_syms_new = syms.len() as isize;
        if n_dims_new < n_dims {
            eprintln!(
                "Expected at least {} new dimensions for integer set",
                n_dims
            );
            exit(ExitCode::DialectError);
        } else if n_syms_new < n_syms {
            eprintln!("Expected at least {} new symbols for integer set", n_syms);
            exit(ExitCode::DialectError);
        }
        let d: Vec<MlirAffineExpr> = dims.iter().map(|d| *d.get()).collect();
        let s: Vec<MlirAffineExpr> = syms.iter().map(|s| *s.get()).collect();
        Self::from(do_unsafe!(mlirIntegerSetReplaceGet(
            *self.get_mut(),
            d.as_ptr(),
            s.as_ptr(),
            n_dims_new,
            n_syms_new,
        )))
    }
}

impl Symbol {
    pub fn new(context: &Context, i: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineSymbolExprGet(*context.get(), i)))
    }

    pub fn from_checked(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_symbol() {
            eprint!("Affine expression given cannot be coerced to a symbol: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self::from(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_position(&self) -> isize {
        do_unsafe!(mlirAffineSymbolExprGetPosition(*self.get()))
    }

    pub fn get_mut(&mut self) -> &mut MlirAffineExpr {
        &mut self.0
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

fn get_dialect(context: &Context) -> Dialect {
    match context.load_dialect("affine") {
        Some(d) => d,
        None => {
            eprintln!("Expected affine dialect to be registered in context");
            exit(ExitCode::DialectError);
        }
    }
}

impl Apply {
    pub fn new(context: &Context, map: Map, indices: &[Value], loc: &Location) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of apply operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context);
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::Apply);
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(indices);
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

/// TODO:   Add support for static basis when MLIR-based tests are fixed.
///         See `tests/lit-tests-mlir/affine_delinearize_index_2.mlir`.
impl DelinearizeIndex {
    pub fn new(context: &Context, index: &Value, dynamic_basis: &[Value], loc: &Location) -> Self {
        if !index.get_type().is_index() {
            eprintln!("Expected index type for index operand of delinearize index operation");
            exit(ExitCode::DialectError);
        }
        if dynamic_basis.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for dynamic basis of delinearize index operation");
            exit(ExitCode::DialectError);
        }
        let n_dynamic = dynamic_basis.len() as i32;
        let t = Index::new(context);
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::DelinearizeIndex);
        let mut operands = vec![index.clone()];
        operands.append(&mut dynamic_basis.to_vec());
        let results: Vec<Type> = (0..n_dynamic).map(|_| t.as_type()).collect();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&operands);
        op_state.add_results(&results);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

#[allow(clippy::too_many_arguments)]
impl DmaStart {
    fn __new(
        context: &Context,
        map_source: Map,
        map_target: Map,
        num_elems: &Value,
        source: &Value,
        target: &Value,
        tagbuf: &Value,
        indices_source: &[Value],
        indices_target: &[Value],
        tag: &Value,
        stride: Option<&Value>,
        num_elems_stride: Option<&Value>,
        loc: &Location,
    ) -> Self {
        Self::check_operands(
            map_source,
            map_target,
            num_elems,
            source,
            target,
            tagbuf,
            indices_source,
            indices_target,
            tag,
            stride,
            num_elems_stride,
        );
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::DmaStart);
        let map_tag = Map::new_identity(context, 1);
        let attr_source =
            NamedMapSource::from(*map_source.as_attribute().get_mut()).as_named_attribute();
        let attr_tag = NamedMapTag::from(*map_tag.as_attribute().get_mut()).as_named_attribute();
        let attr_target =
            NamedMapTarget::from(*map_target.as_attribute().get_mut()).as_named_attribute();
        let mut operands = vec![source.clone()];
        operands.append(&mut indices_source.to_vec());
        operands.push(target.clone());
        operands.append(&mut indices_target.to_vec());
        operands.push(tagbuf.clone());
        operands.push(tag.clone());
        operands.push(num_elems.clone());
        if let Some(n) = stride {
            operands.push(n.clone());
        }
        if let Some(n) = num_elems_stride {
            operands.push(n.clone());
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_source, attr_tag, attr_target]);
        op_state.add_operands(&operands);
        Self::from(*op_state.create_operation().get())
    }

    pub fn new(
        context: &Context,
        map_source: Map,
        map_target: Map,
        num_elems: &Value,
        source: &Value,
        target: &Value,
        tagbuf: &Value,
        indices_source: &[Value],
        indices_target: &[Value],
        tag: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new(
            context,
            map_source,
            map_target,
            num_elems,
            source,
            target,
            tagbuf,
            indices_source,
            indices_target,
            tag,
            None,
            None,
            loc,
        )
    }

    pub fn new_strided(
        context: &Context,
        map_source: Map,
        map_target: Map,
        num_elems: &Value,
        source: &Value,
        target: &Value,
        tagbuf: &Value,
        indices_source: &[Value],
        indices_target: &[Value],
        tag: &Value,
        stride: &Value,
        num_elems_stride: &Value,
        loc: &Location,
    ) -> Self {
        Self::__new(
            context,
            map_source,
            map_target,
            num_elems,
            source,
            target,
            tagbuf,
            indices_source,
            indices_target,
            tag,
            Some(stride),
            Some(num_elems_stride),
            loc,
        )
    }

    fn check_operands(
        map_source: Map,
        map_target: Map,
        num_elems: &Value,
        source: &Value,
        target: &Value,
        tagbuf: &Value,
        indices_source: &[Value],
        indices_target: &[Value],
        tag: &Value,
        stride: Option<&Value>,
        num_elems_stride: Option<&Value>,
    ) -> () {
        let t_source = source.get_type();
        let t_tagbuf = tagbuf.get_type();
        let t_target = target.get_type();
        if !num_elems.get_type().is_index() {
            eprintln!("Expected index type for number of elements operand of dma start operation");
            exit(ExitCode::DialectError);
        }
        if !tag.get_type().is_index() {
            eprintln!("Expected index type for tag operand of dma start operation");
            exit(ExitCode::DialectError);
        }
        if !t_source.is_memref() {
            eprintln!("Expected memory reference type for source operand of dma start operation");
            exit(ExitCode::DialectError);
        }
        if !t_target.is_memref() {
            eprintln!("Expected memory reference type for target operand of dma start operation");
            exit(ExitCode::DialectError);
        }
        if !t_tagbuf.is_memref() {
            eprintln!(
                "Expected memory reference type for tag buffer operand of dma start operation"
            );
            exit(ExitCode::DialectError);
        }
        if indices_source.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for source indices of dma start operation");
            exit(ExitCode::DialectError);
        }
        if indices_target.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for target indices of dma start operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from_type(&t_source);
        let s_tagbuf = Shaped::from_type(&t_tagbuf);
        let s_target = Shaped::from_type(&t_target);
        if s_source.get_element_type() != s_target.get_element_type() {
            eprintln!(
                "Expected matching element type for source and target operands of dma start operation"
            );
            exit(ExitCode::DialectError);
        }
        if !s_tagbuf.get_element_type().is_integer() {
            eprintln!(
                "Expected integer element type for tag buffer operand of dma start operations"
            );
            exit(ExitCode::DialectError);
        }
        let n_source_inputs = map_source.num_inputs();
        let n_source_indices = indices_source.len() as isize;
        if n_source_inputs != n_source_indices {
            eprintln!(
                "Expected number of source indices ({}) to match inputs to source map ({}) of \
                dma start operations",
                n_source_indices, n_source_inputs,
            );
            exit(ExitCode::DialectError);
        }
        let n_target_inputs = map_target.num_inputs();
        let n_target_indices = indices_target.len() as isize;
        if n_target_inputs != n_target_indices {
            eprintln!(
                "Expected number of target indices ({}) to match inputs to target map ({}) of \
                dma start operations",
                n_target_indices, n_target_inputs,
            );
            exit(ExitCode::DialectError);
        }
        if let Some(n) = stride {
            if !n.get_type().is_index() {
                eprintln!("Expected index type for stride of dma start opertation");
                exit(ExitCode::DialectError);
            }
        }
        if let Some(n) = num_elems_stride {
            if !n.get_type().is_index() {
                eprintln!(
                    "Expected index type for number of stride elements of dma start opertation"
                );
                exit(ExitCode::DialectError);
            }
        }
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map_source(&self) -> NamedMapSource {
        let attr_name = StringBacked::from(NamedMapSource::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMapSource::from(*attr.get())
    }

    pub fn get_map_tag(&self) -> NamedMapTag {
        let attr_name = StringBacked::from(NamedMapTag::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMapTag::from(*attr.get())
    }

    pub fn get_map_target(&self) -> NamedMapTarget {
        let attr_name = StringBacked::from(NamedMapTarget::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMapTarget::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl DmaWait {
    pub fn new(
        context: &Context,
        num_elems: &Value,
        tagbuf: &Value,
        tag: &Value,
        loc: &Location,
    ) -> Self {
        if !num_elems.get_type().is_index() {
            eprintln!("Expected index type for number of elements operand of dma wait operation");
            exit(ExitCode::DialectError);
        }
        if !tag.get_type().is_index() {
            eprintln!("Expected index type for tag operand of dma wait operation");
            exit(ExitCode::DialectError);
        }
        let t_tagbuf = tagbuf.get_type();
        if !t_tagbuf.is_memref() {
            eprintln!(
                "Expected memory reference type for tag buffer operand of dma wait operation"
            );
            exit(ExitCode::DialectError);
        }
        let s_tagbuf = Shaped::from_type(&t_tagbuf);
        if !s_tagbuf.get_element_type().is_integer() {
            eprintln!(
                "Expected integer element type for tag buffer operand of dma wait operations"
            );
            exit(ExitCode::DialectError);
        }
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::DmaWait);
        let map_tag = Map::new_identity(context, 1);
        let attr_tag = NamedMapTag::from(*map_tag.as_attribute().get_mut()).as_named_attribute();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_tag]);
        op_state.add_operands(&[tagbuf.clone(), tag.clone(), num_elems.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map_tag(&self) -> NamedMapTag {
        let attr_name = StringBacked::from(NamedMapTag::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMapTag::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl For {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &Context,
        results: &[Type],
        ops_lower_bounds: &[Value],
        ops_upper_bounds: &[Value],
        inits: &[Value],
        map_lower: Map,
        map_upper: Map,
        step: usize,
        loc: &Location,
    ) -> Self {
        Self::check_operands(
            results,
            ops_lower_bounds,
            ops_upper_bounds,
            inits,
            map_lower,
            map_upper,
            step,
        );
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::For);
        let mut operands: Vec<Value> = vec![];
        operands.append(&mut ops_lower_bounds.to_vec());
        operands.append(&mut ops_upper_bounds.to_vec());
        operands.append(&mut inits.to_vec());
        let attr_opseg = OperandSegmentSizes::new(context, &[
            ops_lower_bounds.len() as i32,
            ops_upper_bounds.len() as i32,
            inits.len() as i32,
        ]);
        let n_block_args = 1 + inits.len() as isize;
        let t_index = Index::new(context).as_type();
        let mut t_block = vec![t_index];
        inits.iter().for_each(|v| {
            t_block.push(v.get_type());
        });
        let loc_block: Vec<Location> = (0..n_block_args).map(|_| loc.clone()).collect();
        let mut region = Region::new();
        let mut block = Block::new(n_block_args, &t_block, &loc_block);
        region.append_block(&mut block); // Add empty starter block
        let lower_bounds = LowerBoundsFor::new(map_lower);
        let upper_bounds = UpperBoundsFor::new(map_upper);
        let attr_step = StepFor::new(context, step as i64);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            attr_opseg.as_named_attribute(),
            lower_bounds.as_named_attribute(),
            upper_bounds.as_named_attribute(),
            attr_step.as_named_attribute(),
        ]);
        op_state.add_operands(&operands);
        op_state.add_regions(&[region]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get())
    }

    fn check_operands(
        _results: &[Type],
        ops_lower_bounds: &[Value],
        ops_upper_bounds: &[Value],
        _inits: &[Value],
        map_lower: Map,
        map_upper: Map,
        _step: usize,
    ) -> () {
        if ops_lower_bounds.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for lower bounds operands of for operation");
            exit(ExitCode::DialectError);
        }
        if ops_upper_bounds.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for upper bounds operands of for operation");
            exit(ExitCode::DialectError);
        }
        let n_lower = map_lower.num_inputs();
        let n_upper = map_upper.num_inputs();
        let n_lower_ops = ops_lower_bounds.len() as isize;
        let n_upper_ops = ops_upper_bounds.len() as isize;
        if n_lower != n_lower_ops {
            eprintln!(
                "Expected matching number of lower bound operands ({}) and lower bounds map inputs ({}) \
                for for operation",
                n_lower_ops, n_lower,
            );
            exit(ExitCode::DialectError);
        }
        if n_upper != n_upper_ops {
            eprintln!(
                "Expected matching number of upper bound operands ({}) and upper bounds map inputs ({}) \
                for for operation",
                n_upper_ops, n_upper,
            );
            exit(ExitCode::DialectError);
        }
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_lower_bounds(&self) -> LowerBoundsFor {
        let attr_name = StringBacked::from(LowerBoundsFor::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        LowerBoundsFor::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_step(&self) -> StepFor {
        let attr_name = StringBacked::from(StepFor::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        StepFor::from(*attr.get())
    }

    pub fn get_upper_bounds(&self) -> UpperBoundsFor {
        let attr_name = StringBacked::from(UpperBoundsFor::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        UpperBoundsFor::from(*attr.get())
    }
}

impl If {
    fn new(
        context: &Context,
        results: &[Type],
        inputs: &[Value],
        condition: &Condition,
        regions: &[Region],
        loc: &Location,
    ) -> Self {
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::If);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[condition.as_named_attribute()]);
        op_state.add_operands(inputs);
        op_state.add_regions(regions);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get())
    }

    pub fn new_if(
        context: &Context,
        results: &[Type],
        inputs: &[Value],
        condition: Set,
        loc: &Location,
    ) -> Self {
        let condition_ = Condition::new(condition);
        let mut region_then = Region::new();
        let region_else = Region::new();
        let mut block_then = Block::new_empty();
        region_then.append_block(&mut block_then); // Add empty starter block
        Self::new(
            context,
            results,
            inputs,
            &condition_,
            &[region_then, region_else],
            loc,
        )
    }

    pub fn new_if_else(
        context: &Context,
        results: &[Type],
        inputs: &[Value],
        condition: Set,
        loc: &Location,
    ) -> Self {
        let condition_ = Condition::new(condition);
        let mut region_then = Region::new();
        let mut region_else = Region::new();
        let mut block_then = Block::new_empty();
        let mut block_else = Block::new_empty();
        region_then.append_block(&mut block_then); // Add empty starter block
        region_else.append_block(&mut block_else); // Add empty starter block
        Self::new(
            context,
            results,
            inputs,
            &condition_,
            &[region_then, region_else],
            loc,
        )
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Load {
    pub fn new(t: &Type, map: Map, source: &Value, indices: &[Value], loc: &Location) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of load operation");
            exit(ExitCode::DialectError);
        }
        let t_source = source.get_type();
        if !t_source.is_memref() {
            eprintln!("Expected memref type for source operand of load operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from_type(&t_source);
        if *t != s_source.get_element_type() {
            eprintln!(
                "Expected matching element type for source operand and result of load operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = get_dialect(&context);
        let name = dialect.get_op_name(&Op::Load);
        let mut args = vec![source.clone()];
        args.append(&mut indices.to_vec());
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Max {
    pub fn new(context: &Context, map: Map, indices: &[Value], loc: &Location) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of max operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context).as_type();
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::Max);
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(indices);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Min {
    pub fn new(context: &Context, map: Map, indices: &[Value], loc: &Location) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of min operation");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context).as_type();
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::Min);
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(indices);
        op_state.add_results(&[t]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Parallel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &Context,
        results: &[Type],
        maps_lower: &[Map],
        maps_upper: &[Map],
        ops_lower_bounds: &[Value],
        ops_upper_bounds: &[Value],
        steps: Option<&[i64]>,
        ops_reductions: &[ReductionOp],
        loc: &Location,
    ) -> Self {
        Self::check_operands(
            results,
            ops_lower_bounds,
            ops_upper_bounds,
            ops_reductions,
            maps_lower,
            maps_upper,
        );
        let (map_lower, map_upper, groups_lower, groups_upper) =
            Self::check_maps(context, maps_lower, maps_upper);
        let steps_ = Self::check_steps(&groups_lower, &groups_upper, steps);
        let t_i32 = IntegerType::new(context, 32).as_type();
        let t_index = Index::new(context).as_type();
        let s_lower = ShapeImpl::from(vec![groups_lower.len() as i64]);
        let s_upper = ShapeImpl::from(vec![groups_upper.len() as i64]);
        let t_tnsr_lower = RankedTensor::new(&s_lower, &t_i32).as_shaped();
        let t_tnsr_upper = RankedTensor::new(&s_upper, &t_i32).as_shaped();
        let attr_lower = LowerBoundsParallel::new(map_lower).as_named_attribute();
        let attr_upper = UpperBoundsParallel::new(map_upper).as_named_attribute();
        let attr_lower_groups =
            LowerBoundGroups::new(&t_tnsr_lower, &groups_lower).as_named_attribute();
        let attr_upper_groups =
            UpperBoundGroups::new(&t_tnsr_upper, &groups_upper).as_named_attribute();
        let attr_reductions = Reductions::new(context, ops_reductions).as_named_attribute();
        let attr_steps = StepsParallel::new(context, &steps_).as_named_attribute();
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::Parallel);
        let mut operands: Vec<Value> = vec![];
        operands.append(&mut ops_lower_bounds.to_vec());
        operands.append(&mut ops_upper_bounds.to_vec());
        let mut region = Region::new();
        let n_block = steps_.len();
        let t_block: Vec<Type> = (0..n_block).map(|_| t_index.clone()).collect();
        let loc_block: Vec<Location> = (0..n_block).map(|_| loc.clone()).collect();
        let mut block = Block::new(n_block as isize, &t_block, &loc_block);
        region.append_block(&mut block);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_lower, attr_upper, attr_lower_groups, attr_upper_groups]);
        op_state.add_operands(&operands);
        op_state.add_regions(&[region]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        let mut op = op_state.create_operation();
        op.set_named_attribute_discardable(&attr_reductions);
        op.set_named_attribute_discardable(&attr_steps);
        Self::from(*op.get_mut())
    }

    /// Check map parameters and return the map groups and combined lower and upper maps across groups.
    fn check_maps(
        context: &Context,
        maps_lower: &[Map],
        maps_upper: &[Map],
    ) -> (Map, Map, Vec<i32>, Vec<i32>) {
        let Some(map_lower_0) = maps_lower.first() else {
            eprintln!("Expected at least one map for lower bounds of parallel operation");
            exit(ExitCode::DialectError);
        };
        let Some(map_upper_0) = maps_upper.first() else {
            eprintln!("Expected at least one map for upper bounds of parallel operation");
            exit(ExitCode::DialectError);
        };
        let n_dims_lower = map_lower_0.num_dims();
        let n_dims_upper = map_upper_0.num_dims();
        let n_syms_lower = map_lower_0.num_symbols();
        let n_syms_upper = map_upper_0.num_symbols();
        if maps_lower.iter().any(|map| map.num_dims() != n_dims_lower) {
            eprintln!(
                "Expected matching number of dimensions lower bounds maps of parallel operation"
            );
            exit(ExitCode::DialectError);
        }
        if maps_upper.iter().any(|map| map.num_dims() != n_dims_upper) {
            eprintln!(
                "Expected matching number of dimensions upper bounds maps of parallel operation"
            );
            exit(ExitCode::DialectError);
        }
        if maps_lower
            .iter()
            .any(|map| map.num_symbols() != n_syms_lower)
        {
            eprintln!(
                "Expected matching number of symbols lower bounds maps of parallel operation"
            );
            exit(ExitCode::DialectError);
        }
        if maps_upper
            .iter()
            .any(|map| map.num_symbols() != n_syms_upper)
        {
            eprintln!(
                "Expected matching number of symbols upper bounds maps of parallel operation"
            );
            exit(ExitCode::DialectError);
        }
        let mut groups_lower: Vec<i32> = vec![];
        let mut groups_upper: Vec<i32> = vec![];
        let mut results_lower: Vec<Expr> = vec![];
        let mut results_upper: Vec<Expr> = vec![];
        for map in maps_lower.iter() {
            let n_results = map.num_results();
            (0..n_results).for_each(|i| results_lower.push(map.get_result(i)));
            groups_lower.push(n_results as i32);
        }
        for map in maps_upper.iter() {
            let n_results = map.num_results();
            (0..n_results).for_each(|i| results_upper.push(map.get_result(i)));
            groups_upper.push(n_results as i32);
        }
        let map_lower = Map::new_results(context, n_dims_lower, n_syms_lower, &results_lower);
        let map_upper = Map::new_results(context, n_dims_upper, n_syms_upper, &results_upper);
        (map_lower, map_upper, groups_lower, groups_upper)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_operands(
        _results: &[Type],
        ops_lower_bounds: &[Value],
        ops_upper_bounds: &[Value],
        _ops_reductions: &[ReductionOp], // TODO: Validate against arith binary ops?
        map_lower: &[Map],
        map_upper: &[Map],
    ) -> () {
        if ops_lower_bounds.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for lower bounds operands of parallel operation");
            exit(ExitCode::DialectError);
        }
        if ops_upper_bounds.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for upper bounds operands of parallel operation");
            exit(ExitCode::DialectError);
        }
        let n_lower_ops = ops_lower_bounds.len() as isize;
        for map in map_lower.iter() {
            let n_lower = map.num_inputs();
            if n_lower != n_lower_ops {
                eprintln!(
                    "Expected matching number of lower bound operands ({}) and lower bounds map inputs \
                    ({}) for parallel operation",
                    n_lower_ops, n_lower,
                );
                exit(ExitCode::DialectError);
            }
        }
        let n_upper_ops = ops_upper_bounds.len() as isize;
        for map in map_upper.iter() {
            let n_upper = map.num_inputs();
            if n_upper != n_upper_ops {
                eprintln!(
                    "Expected matching number of upper bound operands ({}) and upper bounds map inputs \
                    ({}) for parallel operation",
                    n_upper_ops, n_upper,
                );
                exit(ExitCode::DialectError);
            }
        }
    }

    /// Check if the number of given steps are valid for the given upper and lower bounds.
    /// If no steps were passed in by the user, return the default step(s) based on the bounds.
    fn check_steps(groups_lower: &[i32], groups_upper: &[i32], steps: Option<&[i64]>) -> Vec<i64> {
        let n_lower = groups_lower.len();
        let n_upper = groups_upper.len();
        if n_lower != n_upper {
            eprintln!(
                "Expected matching number of lower ({}) and upper ({}) groups of parallel operation",
                n_lower, n_upper,
            );
            exit(ExitCode::DialectError);
        }
        match steps {
            Some(slice) => {
                let n_slice = slice.len();
                if n_slice != n_lower {
                    eprintln!(
                        "Expected matching number of steps ({}) and lower/upper groups ({}) \
                        of parallel operation",
                        n_slice, n_lower,
                    );
                }
                slice.to_vec()
            }
            None => (0..groups_lower.len()).map(|_| 1).collect::<Vec<i64>>(),
        }
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_lower_bounds(&self) -> LowerBoundsParallel {
        let attr_name = StringBacked::from(LowerBoundsParallel::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        LowerBoundsParallel::from(*attr.get())
    }

    pub fn get_lower_bounds_groups(&self) -> LowerBoundGroups {
        let attr_name = StringBacked::from(LowerBoundGroups::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        LowerBoundGroups::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_reductions(&self) -> Reductions {
        let attr_name = StringBacked::from(Reductions::get_name());
        let attr = self
            .as_operation()
            .get_attribute_discardable(&attr_name.as_string_ref());
        Reductions::from(*attr.get())
    }

    pub fn get_steps(&self) -> StepsParallel {
        let attr_name = StringBacked::from(StepsParallel::get_name());
        let attr = self
            .as_operation()
            .get_attribute_discardable(&attr_name.as_string_ref());
        StepsParallel::from(*attr.get())
    }

    pub fn get_upper_bounds(&self) -> UpperBoundsParallel {
        let attr_name = StringBacked::from(UpperBoundsParallel::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        UpperBoundsParallel::from(*attr.get())
    }

    pub fn get_upper_bounds_groups(&self) -> UpperBoundGroups {
        let attr_name = StringBacked::from(UpperBoundGroups::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        UpperBoundGroups::from(*attr.get())
    }
}

impl Store {
    pub fn new(
        context: &Context,
        map: Map,
        value: &Value,
        target: &Value,
        indices: &[Value],
        loc: &Location,
    ) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of store operation");
            exit(ExitCode::DialectError);
        }
        let t = value.get_type();
        let t_target = target.get_type();
        if !t_target.is_memref() {
            eprintln!("Expected memref type for target operand of store operation");
            exit(ExitCode::DialectError);
        }
        let s_target = Shaped::from_type(&t_target);
        if t != s_target.get_element_type() {
            eprintln!(
                "Expected matching element type for target operand and result of store operation"
            );
            exit(ExitCode::DialectError);
        }
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::Store);
        let mut args = vec![value.clone(), target.clone()];
        args.append(&mut indices.to_vec());
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&args);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl VectorLoad {
    pub fn new(t: &Vector, map: Map, source: &Value, indices: &[Value], loc: &Location) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of vector load operation");
            exit(ExitCode::DialectError);
        }
        let t_source = source.get_type();
        if !t_source.is_memref() {
            eprintln!("Expected memref type for source operand of vector load operation");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_source = Shaped::from_type(&t_source);
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!(
                "Expected matching element type for source operand and result of vector load operation"
            );
            exit(ExitCode::DialectError);
        }
        let n = s.num_elements().unwrap_or(-1);
        let n_source = s_source.num_elements().unwrap_or(-1);
        if n > n_source {
            eprintln!(
                "Expected number of elements less than or equal to source operand ({}) for result ({}) \
                of vector load operation",
                n_source, n
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = get_dialect(&context);
        let name = dialect.get_op_name(&Op::VectorLoad);
        let mut args = vec![source.clone()];
        args.append(&mut indices.to_vec());
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl VectorStore {
    pub fn new(
        context: &Context,
        map: Map,
        value: &Value,
        source: &Value,
        indices: &[Value],
        loc: &Location,
    ) -> Self {
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for indices of vector store operation");
            exit(ExitCode::DialectError);
        }
        let t = value.get_type();
        if !t.is_vector() {
            eprintln!("Expected vector type for value operand of vector store operation");
            exit(ExitCode::DialectError);
        }
        let t_source = source.get_type();
        if !t_source.is_memref() {
            eprintln!("Expected memref type for source operand of vector store operation");
            exit(ExitCode::DialectError);
        }
        let s = Shaped::from_type(&t);
        let s_source = Shaped::from_type(&t_source);
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!(
                "Expected matching element type for source operand and result of vector store operation"
            );
            exit(ExitCode::DialectError);
        }
        let n = s.num_elements().unwrap_or(-1);
        let n_source = s_source.num_elements().unwrap_or(-1);
        if n > n_source {
            eprintln!(
                "Expected number of elements less than or equal to source operand ({}) for result ({}) \
                of vector store operation",
                n_source, n
            );
            exit(ExitCode::DialectError);
        }
        let dialect = get_dialect(context);
        let name = dialect.get_op_name(&Op::VectorStore);
        let mut args = vec![value.clone(), source.clone()];
        args.append(&mut indices.to_vec());
        let attr = NamedMap::from(*map.as_attribute().get_mut());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_operands(&args);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_map(&self) -> NamedMap {
        let attr_name = StringBacked::from(NamedMap::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NamedMap::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Yield {
    fn __new(
        values: &[Value],
        parent: &MlirOperation,
        parent_op: &Op,
        dialect: &Dialect,
        loc: &Location,
    ) -> Self {
        let name = dialect.get_op_name(&Op::Yield);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(values);
        Self::from(*op_state.create_operation().get(), *parent, *parent_op)
    }

    fn new(values: &[Value], parent_: &dyn IOperation, loc: &Location) -> Self {
        let parent = parent_.as_operation();
        let context = parent.get_context();
        let dialect = get_dialect(&context);
        if parent_.get_dialect() != dialect {
            eprintln!("Expected parent operation is from affine dialect");
            exit(ExitCode::DialectError);
        }
        let parent_op = match parent_.get_op().get_name() {
            "for" => Op::For,
            "if" => Op::If,
            "parallel" => Op::Parallel,
            _ => {
                eprintln!(
                    "Expected parent operation of yield is a affine for, if, or parallel operation"
                );
                exit(ExitCode::DialectError);
            }
        };
        let t: Vec<Type> = values.iter().map(|v| v.get_type()).collect();
        let n_inputs = t.len() as isize;
        let n_results = parent.num_results();
        if n_inputs != n_results {
            eprintln!(
                "Expected matching number of inputs ({}) and parent operation results ({}) of \
                yield operation",
                n_inputs, n_results,
            );
            exit(ExitCode::DialectError);
        }
        if t.iter()
            .enumerate()
            .any(|(i, t_)| *t_ != parent.get_result(i as isize).get_type())
        {
            eprintln!(
                "Expected matching types for inputs and parent operation results of yield operation",
            );
            exit(ExitCode::DialectError);
        }
        Self::__new(values, parent.get(), &parent_op, &dialect, loc)
    }

    pub fn new_for(values: &[Value], parent: &For, loc: &Location) -> Self {
        Self::new(values, parent, loc)
    }

    pub fn new_if(values: &[Value], parent: &If, loc: &Location) -> Self {
        Self::new(values, parent, loc)
    }

    pub fn new_parallel(values: &[Value], parent: &Parallel, loc: &Location) -> Self {
        Self::new(values, parent, loc)
    }

    pub fn from(op: MlirOperation, parent: MlirOperation, parent_op: Op) -> Self {
        Self(op, parent, parent_op)
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

    pub fn get_result(&self, pos: isize) -> Value {
        self.as_operation().get_result(pos)
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl From<MlirOperation> for Apply {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Apply {
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
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Apply.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Apply
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<(MlirAffineExpr, BinOp)> for Binary {
    fn from((expr, op): (MlirAffineExpr, BinOp)) -> Self {
        Self(expr, op)
    }
}

impl IExpr for Binary {
    fn as_expr(&self) -> Expr {
        Expr::from(*self.get())
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

SpecializedAttribute!("condition" = impl NamedAffineSet for Condition {});

impl From<MlirAffineExpr> for Constant {
    fn from(expr: MlirAffineExpr) -> Self {
        Self(expr)
    }
}

impl IExpr for Constant {
    fn as_expr(&self) -> Expr {
        Expr::from(*self.get())
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl From<MlirOperation> for DelinearizeIndex {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DelinearizeIndex {
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
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::DelinearizeIndex.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::DelinearizeIndex
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirAffineExpr> for Dim {
    fn from(expr: MlirAffineExpr) -> Self {
        Self(expr)
    }
}

impl IExpr for Dim {
    fn as_expr(&self) -> Expr {
        Expr::from(*self.get())
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl From<MlirOperation> for DmaStart {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DmaStart {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
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
        Op::DmaStart.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::DmaStart
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for DmaWait {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for DmaWait {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
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
        Op::DmaWait.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::DmaWait
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirAffineExpr> for Expr {
    fn from(expr: MlirAffineExpr) -> Self {
        Self(expr)
    }
}

impl IExpr for Expr {
    fn as_expr(&self) -> Expr {
        *self
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl From<MlirOperation> for For {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for For {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::LoopLikeOpInterface,
            Interface::RegionBranchOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::For.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::For
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::AutomaticAllocationScope,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

impl From<MlirOperation> for If {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for If {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::RegionBranchOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::If.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::If
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::NoRegionArguments,
            Trait::RecursiveMemoryEffects,
            Trait::RecursivelySpeculatableImplTrait,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

impl cmp::PartialEq for dyn IExpr {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAffineExprEqual(*self.get(), *rhs.get()))
    }
}

impl From<MlirOperation> for Load {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Load {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::AffineMapAccessInterface,
            Interface::AffineReadOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Load.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Load
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

SpecializedAttribute!("lowerBoundMap" = impl NamedAffineMap for LowerBoundsFor {});

SpecializedAttribute!("lowerBoundsMap" = impl NamedAffineMap for LowerBoundsParallel {});

SpecializedAttribute!("lowerBoundsGroups" = impl NamedI32DenseElements for LowerBoundGroups {});

impl From<MlirAffineMap> for Map {
    fn from(attr: MlirAffineMap) -> Self {
        Self(attr)
    }
}

impl From<Attribute> for Map {
    fn from(attr: Attribute) -> Self {
        Self::from(&attr)
    }
}

impl From<&Attribute> for Map {
    fn from(attr: &Attribute) -> Self {
        Self::from_attribute(attr)
    }
}

impl From<MlirAttribute> for Map {
    fn from(attr: MlirAttribute) -> Self {
        Self::from_attribute(&Attribute::from(attr))
    }
}

impl cmp::PartialEq for Map {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAffineMapEqual(*self.get(), *rhs.get()))
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
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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
        Op::Max.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Max
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
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
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
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
        Op::Min.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Min
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

SpecializedAttribute!("map" = impl NamedAffineMap for NamedMap {});

SpecializedAttribute!("src_map" = impl NamedAffineMap for NamedMapSource {});

SpecializedAttribute!("tag_map" = impl NamedAffineMap for NamedMapTag {});

SpecializedAttribute!("dst_map" = impl NamedAffineMap for NamedMapTarget {});

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for Parallel {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Parallel {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::LoopLikeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Parallel.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Parallel
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AutomaticAllocationScope,
            Trait::MemRefsNormalizable,
            Trait::RecursiveMemoryEffects,
            Trait::RecursivelySpeculatableImplTrait,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

SpecializedAttribute!("reductions" = impl NamedArrayOfIntegers for Reductions {});

impl From<MlirIntegerSet> for Set {
    fn from(set: MlirIntegerSet) -> Self {
        Self(set)
    }
}

impl cmp::PartialEq for Set {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirIntegerSetEqual(*self.get(), *rhs.get()))
    }
}

SpecializedAttribute!("step" = impl NamedIndex for StepFor {});

SpecializedAttribute!("steps" = impl NamedArrayOfIntegers for StepsParallel {});

impl From<MlirOperation> for Store {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Store {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::AffineMapAccessInterface,
            Interface::AffineWriteOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Store.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Store
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl From<MlirAffineExpr> for Symbol {
    fn from(expr: MlirAffineExpr) -> Self {
        Self(expr)
    }
}

impl IExpr for Symbol {
    fn as_expr(&self) -> Expr {
        Expr::from(*self.get())
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

SpecializedAttribute!("upperBoundMap" = impl NamedAffineMap for UpperBoundsFor {});

SpecializedAttribute!("upperBoundsMap" = impl NamedAffineMap for UpperBoundsParallel {});

SpecializedAttribute!("upperBoundsGroups" = impl NamedI32DenseElements for UpperBoundGroups {});

impl From<MlirOperation> for VectorLoad {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for VectorLoad {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::AffineMapAccessInterface,
            Interface::AffineReadOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::VectorLoad.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::VectorLoad
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl From<MlirOperation> for VectorStore {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for VectorStore {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        get_dialect(&self.as_operation().get_context())
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::AffineMapAccessInterface,
            Interface::AffineWriteOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::VectorStore.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::VectorStore
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl IOperation for Yield {
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

    fn get_op(&self) -> OpRef {
        &Op::Yield
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::MemRefsNormalizable,
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

impl cmp::PartialEq for Yield {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
            && self.get_parent_op() == rhs.get_parent_op()
            && Operation::from(*self.get_parent()) == Operation::from(*rhs.get_parent())
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            BinOp::Add => "+",
            BinOp::Mod => "%",
            BinOp::Mul => "*",
            BinOp::CeilDiv => "ceildiv",
            BinOp::FloorDiv => "floordiv",
        })
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Apply => "ApplyOp",
            Op::DelinearizeIndex => "DelinearizeIndexOp",
            Op::DmaStart => "DmaStartOp",
            Op::DmaWait => "DmaWaitOp",
            Op::For => "ForOp",
            Op::If => "IfOp",
            Op::LinearizeIndex => "LinearizeIndexOp",
            Op::Load => "LoadOp",
            Op::Max => "MaxOp",
            Op::Min => "MinOp",
            Op::Parallel => "ParallelOp",
            Op::Prefetch => "PrefetchOp",
            Op::Store => "StoreOp",
            Op::VectorLoad => "VectorLoadOp",
            Op::VectorStore => "VectorStoreOp",
            Op::Yield => "YieldOp",
        })
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}
