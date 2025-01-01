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

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::specialized::NamedAffineMap;
use attributes::specialized::NamedAffineSet;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedIndex;
use dialects::IOp;
use dialects::IOperation;
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
pub struct LowerBounds(MlirAttribute);

#[derive(Clone)]
pub struct NamedMap(MlirAttribute);

#[derive(Clone)]
pub struct Step(MlirAttribute);

#[derive(Clone)]
pub struct UpperBounds(MlirAttribute);

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

impl LowerBounds {
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

impl Step {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl UpperBounds {
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
        Self::from(
            do_unsafe!(mlirAffineAddExprGet(*lhs.get(), *rhs.get())),
            BinOp::Add,
        )
    }

    pub fn new_mod(lhs: Expr, rhs: Expr) -> Self {
        Self::from(
            do_unsafe!(mlirAffineModExprGet(*lhs.get(), *rhs.get())),
            BinOp::Mod,
        )
    }

    pub fn new_mul(lhs: Expr, rhs: Expr) -> Self {
        Self::from(
            do_unsafe!(mlirAffineMulExprGet(*lhs.get(), *rhs.get())),
            BinOp::Mul,
        )
    }

    pub fn new_ceil_div(lhs: Expr, rhs: Expr) -> Self {
        Self::from(
            do_unsafe!(mlirAffineCeilDivExprGet(*lhs.get(), *rhs.get())),
            BinOp::CeilDiv,
        )
    }

    pub fn new_floor_div(lhs: Expr, rhs: Expr) -> Self {
        Self::from(
            do_unsafe!(mlirAffineFloorDivExprGet(*lhs.get(), *rhs.get())),
            BinOp::FloorDiv,
        )
    }

    pub fn new_sub(lhs: Expr, rhs_: Expr) -> Self {
        let context = lhs.get_context();
        let cn1 = Constant::new(&context, -1).as_expr();
        let rhs = Self::new_mul(rhs_, cn1).as_expr();
        Self::from(
            do_unsafe!(mlirAffineAddExprGet(*lhs.get(), *rhs.get())),
            BinOp::Add,
        )
    }

    pub fn from(expr: MlirAffineExpr, op: BinOp) -> Self {
        let expr_ = Expr::from(expr);
        // NOTE: Incoming expression may have been internally simplified to a constant (e.g., const * -1).
        if !expr_.is_binary() && !expr_.is_constant() {
            eprint!("Affine expression given cannot be coerced to a binary operation: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Self(expr, op)
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
            Some(Binary::from(*self.get(), op))
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
        step: &Step,
        loc: &Location,
    ) -> Self {
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
        let lower_bounds = LowerBounds::new(map_lower);
        let upper_bounds = UpperBounds::new(map_upper);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            attr_opseg.as_named_attribute(),
            lower_bounds.as_named_attribute(),
            upper_bounds.as_named_attribute(),
            step.as_named_attribute(),
        ]);
        op_state.add_operands(&operands);
        op_state.add_regions(&[region]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
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
        let dialect = context.get_dialect_tensor();
        if parent_.get_dialect() != dialect {
            eprintln!("Expected parent operation is from affine dialect");
            exit(ExitCode::DialectError);
        }
        let parent_op = match parent_.get_op().get_name() {
            "for" => Op::For,
            "if" => Op::If,
            _ => {
                eprintln!("Expected parent operation of yield is a affine for or if operation");
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
        Self::__new(values, parent.get(), &Op::For, &parent.get_dialect(), loc)
    }

    pub fn new_if(values: &[Value], parent: &If, loc: &Location) -> Self {
        Self::__new(values, parent.get(), &Op::If, &parent.get_dialect(), loc)
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

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Apply
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
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

impl From<MlirAttribute> for Condition {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for Condition {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for Condition {
    fn get_name() -> &'static str {
        "condition"
    }
}

impl NamedAffineSet for Condition {}

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

    fn get_op(&self) -> &'static dyn IOp {
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

    fn get_op(&self) -> &'static dyn IOp {
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

    fn get_op(&self) -> &'static dyn IOp {
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

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Load
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl From<MlirAttribute> for LowerBounds {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for LowerBounds {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for LowerBounds {
    fn get_name() -> &'static str {
        "lowerBoundMap"
    }
}

impl NamedAffineMap for LowerBounds {}

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

impl From<MlirAttribute> for NamedMap {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for NamedMap {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for NamedMap {
    fn get_name() -> &'static str {
        "map"
    }
}

impl NamedAffineMap for NamedMap {}

impl cmp::PartialEq for Map {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAffineMapEqual(*self.get(), *rhs.get()))
    }
}

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

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

impl From<MlirAttribute> for Step {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for Step {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for Step {
    fn get_name() -> &'static str {
        "step"
    }
}

impl NamedIndex for Step {}

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

    fn get_op(&self) -> &'static dyn IOp {
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

impl From<MlirAttribute> for UpperBounds {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for UpperBounds {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for UpperBounds {
    fn get_name() -> &'static str {
        "upperBoundMap"
    }
}

impl NamedAffineMap for UpperBounds {}

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

    fn get_op(&self) -> &'static dyn IOp {
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

    fn get_op(&self) -> &'static dyn IOp {
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

    fn get_op(&self) -> &'static dyn IOp {
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
