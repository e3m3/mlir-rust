// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirAffineAddExprGet;
use mlir::mlirAffineBinaryOpExprGetLHS;
use mlir::mlirAffineBinaryOpExprGetRHS;
use mlir::mlirAffineCeilDivExprGet;
use mlir::mlirAffineConstantExprGet;
use mlir::mlirAffineConstantExprGetValue;
use mlir::mlirAffineDimExprGet;
use mlir::mlirAffineDimExprGetPosition;
use mlir::mlirAffineExprCompose;
use mlir::mlirAffineExprDump;
use mlir::mlirAffineExprEqual;
use mlir::mlirAffineExprGetContext;
use mlir::mlirAffineExprGetLargestKnownDivisor;
use mlir::mlirAffineExprIsAAdd;
use mlir::mlirAffineExprIsABinary;
use mlir::mlirAffineExprIsACeilDiv;
use mlir::mlirAffineExprIsAConstant;
use mlir::mlirAffineExprIsADim;
use mlir::mlirAffineExprIsAFloorDiv;
use mlir::mlirAffineExprIsAMod;
use mlir::mlirAffineExprIsAMul;
use mlir::mlirAffineExprIsASymbol;
use mlir::mlirAffineExprIsFunctionOfDim;
use mlir::mlirAffineExprIsMultipleOf;
use mlir::mlirAffineExprIsPureAffine;
use mlir::mlirAffineExprIsSymbolicOrConstant;
use mlir::mlirAffineFloorDivExprGet;
use mlir::mlirAffineMapAttrGet;
use mlir::mlirAffineMapAttrGetValue;
use mlir::mlirAffineMapAttrGetTypeID;
use mlir::mlirAffineMapConstantGet;
use mlir::mlirAffineMapDump;
use mlir::mlirAffineMapEqual;
use mlir::mlirAffineMapEmptyGet;
use mlir::mlirAffineMapGet;
use mlir::mlirAffineMapGetContext;
use mlir::mlirAffineMapGetMajorSubMap;
use mlir::mlirAffineMapGetMinorSubMap;
use mlir::mlirAffineMapGetNumDims;
use mlir::mlirAffineMapGetNumInputs;
use mlir::mlirAffineMapGetNumResults;
use mlir::mlirAffineMapGetNumSymbols;
use mlir::mlirAffineMapGetResult;
use mlir::mlirAffineMapGetSingleConstantResult;
use mlir::mlirAffineMapGetSubMap;
use mlir::mlirAffineMapIsEmpty;
use mlir::mlirAffineMapIsIdentity;
use mlir::mlirAffineMapIsMinorIdentity;
use mlir::mlirAffineMapIsPermutation;
use mlir::mlirAffineMapIsProjectedPermutation;
use mlir::mlirAffineMapIsSingleConstant;
use mlir::mlirAffineMapMinorIdentityGet;
use mlir::mlirAffineMapMultiDimIdentityGet;
use mlir::mlirAffineMapPermutationGet;
use mlir::mlirAffineMapReplace;
use mlir::mlirAffineMapZeroResultGet;
use mlir::mlirAffineModExprGet;
use mlir::mlirAffineMulExprGet;
use mlir::mlirAffineSymbolExprGet;
use mlir::mlirAffineSymbolExprGetPosition;
use mlir::MlirAttribute;
use mlir::MlirAffineExpr;
use mlir::MlirAffineMap;

use std::cmp;
use std::ffi::c_uint;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::TypeID;

pub trait AffineExpr {
    fn as_expr(&self) -> Expr;
    fn get(&self) -> &MlirAffineExpr;
}

#[derive(Copy,Clone,PartialEq)]
pub enum BinOp {
    Add,
    Mod,
    Mul,
    CeilDiv,
    FloorDiv,
}

#[derive(Clone)]
pub struct Binary(MlirAffineExpr, BinOp);

#[derive(Clone)]
pub struct Constant(MlirAffineExpr);

#[derive(Clone)]
pub struct Dim(MlirAffineExpr);

#[derive(Clone)]
pub struct Expr(MlirAffineExpr);

#[derive(Clone)]
pub struct Map(MlirAffineMap);

#[derive(Clone)]
pub struct Symbol(MlirAffineExpr);

impl Binary {
    pub fn new_add(lhs: &Expr, rhs: &Expr) -> Self {
        Self::from(do_unsafe!(mlirAffineAddExprGet(*lhs.get(), *rhs.get())), BinOp::Add)
    }

    pub fn new_mod(lhs: &Expr, rhs: &Expr) -> Self {
        Self::from(do_unsafe!(mlirAffineModExprGet(*lhs.get(), *rhs.get())), BinOp::Mod)
    }

    pub fn new_mul(lhs: &Expr, rhs: &Expr) -> Self {
        Self::from(do_unsafe!(mlirAffineMulExprGet(*lhs.get(), *rhs.get())), BinOp::Mul)
    }

    pub fn new_ceil_div(lhs: &Expr, rhs: &Expr) -> Self {
        Self::from(do_unsafe!(mlirAffineCeilDivExprGet(*lhs.get(), *rhs.get())), BinOp::CeilDiv)
    }

    pub fn new_floor_div(lhs: &Expr, rhs: &Expr) -> Self {
        Self::from(do_unsafe!(mlirAffineFloorDivExprGet(*lhs.get(), *rhs.get())), BinOp::FloorDiv)
    }

    pub fn from(expr: MlirAffineExpr, op: BinOp) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_binary() {
            eprint!("Affine expression given cannot be coerced to a binary operation: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Binary(expr, op)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_op(&self) -> BinOp {
        self.1
    }

    pub fn get_lhs(&self) -> Expr {
        Expr::from(do_unsafe!(mlirAffineBinaryOpExprGetLHS(self.0)))
    }

    pub fn get_rhs(&self) -> Expr {
        Expr::from(do_unsafe!(mlirAffineBinaryOpExprGetRHS(self.0)))
    }
}

impl Constant {
    pub fn new(context: &Context, constant: i64) -> Self {
        Self::from(do_unsafe!(mlirAffineConstantExprGet(*context.get(), constant)))
    }

    pub fn from(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_constant() {
            eprint!("Affine expression given cannot be coerced to a constant: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Constant(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_value(&self) -> i64 {
        do_unsafe!(mlirAffineConstantExprGetValue(self.0))
    }
}

impl Dim {
    pub fn new(context: &Context, i: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineDimExprGet(*context.get(), i)))
    }

    pub fn from(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_dim() {
            eprint!("Affine expression given cannot be coerced to a dimension: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Dim(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_position(&self) -> isize {
        do_unsafe!(mlirAffineDimExprGetPosition(self.0))
    }
}

impl Expr {
    pub fn from(expr: MlirAffineExpr) -> Self {
        Expr(expr)
    }

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
            Some(Binary::from(self.0, op))
        } else {
            None
        }
    }

    pub fn as_constant(&self) -> Option<Constant> {
        if self.is_constant() {
            Some(Constant::from(self.0))
        } else {
            None
        }
    }

    pub fn as_dim(&self) -> Option<Dim> {
        if self.is_dim() {
            Some(Dim::from(self.0))
        } else {
            None
        }
    }

    pub fn as_symbol(&self) -> Option<Symbol> {
        if self.is_symbol() {
            Some(Symbol::from(self.0))
        } else {
            None
        }
    }

    pub fn compose(&self, map: &Map) -> Self {
        Self::from(do_unsafe!(mlirAffineExprCompose(self.0, *map.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirAffineExprDump(self.0))
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirAffineExprGetContext(self.0)))
    }

    pub fn get_largest_divisor(&self) -> i64 {
        do_unsafe!(mlirAffineExprGetLargestKnownDivisor(self.0))
    }

    pub fn is_add(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAAdd(self.0))
    }

    pub fn is_mod(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAMod(self.0))
    }

    pub fn is_mul(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAMul(self.0))
    }

    pub fn is_ceil_div(&self) -> bool {
        do_unsafe!(mlirAffineExprIsACeilDiv(self.0))
    }

    pub fn is_floor_div(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAFloorDiv(self.0))
    }

    pub fn is_binary(&self) -> bool {
        do_unsafe!(mlirAffineExprIsABinary(self.0))
    }

    pub fn is_constant(&self) -> bool {
        do_unsafe!(mlirAffineExprIsAConstant(self.0))
    }

    pub fn is_dim(&self) -> bool {
        do_unsafe!(mlirAffineExprIsADim(self.0))
    }

    pub fn is_function_of_dim(&self, pos: isize) -> bool {
        do_unsafe!(mlirAffineExprIsFunctionOfDim(self.0, pos))
    }

    pub fn is_multiple_of(&self, factor: i64) -> bool {
        do_unsafe!(mlirAffineExprIsMultipleOf(self.0, factor))
    }

    pub fn is_pure(&self) -> bool {
        do_unsafe!(mlirAffineExprIsPureAffine(self.0))
    }

    pub fn is_symbol(&self) -> bool {
        do_unsafe!(mlirAffineExprIsASymbol(self.0))
    }

    pub fn is_symbolic_or_constant(&self) -> bool {
        do_unsafe!(mlirAffineExprIsSymbolicOrConstant(self.0))
    }
}

impl Map {
    pub fn new(context: &Context) -> Self {
        Self::from(do_unsafe!(mlirAffineMapEmptyGet(*context.get())))
    }

    pub fn new_constant(context: &Context, constant: i64) -> Self {
        Self::from(do_unsafe!(mlirAffineMapConstantGet(*context.get(), constant)))
    }

    pub fn new_minor_identity(context: &Context, num_dims: isize, num_results: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapMinorIdentityGet(*context.get(), num_dims, num_results)))
    }

    pub fn new_identity(context: &Context, num_dims: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapMultiDimIdentityGet(*context.get(), num_dims)))
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
        Self::from(do_unsafe!(mlirAffineMapZeroResultGet(*context.get(), num_dims, num_syms)))
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
        Attribute::from(do_unsafe!(mlirAffineMapAttrGet(self.0)))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirAffineMapDump(self.0))
    }

    pub fn get(&self) -> &MlirAffineMap {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirAffineMapGetContext(self.0)))
    }

    pub fn get_major_sub_map(&self, num_results: isize) -> Option<Self> {
        if num_results <= 0 {
            None
        } else if num_results >= self.num_results() {
            Some(self.clone())
        } else {
            Some(Self::from(do_unsafe!(mlirAffineMapGetMajorSubMap(self.0, num_results))))
        }
    }

    pub fn get_minor_sub_map(&self, num_results: isize) -> Option<Self> {
        if num_results <= 0 {
            None
        } else if num_results >= self.num_results() {
            Some(self.clone())
        } else {
            Some(Self::from(do_unsafe!(mlirAffineMapGetMinorSubMap(self.0, num_results))))
        }
    }

    pub fn get_result(&self, i: isize) -> Expr {
        if i >= self.num_results() || i < 0 {
            eprint!("Index '{}' out of bounds for affine map results: ", i);
            self.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Expr::from(do_unsafe!(mlirAffineMapGetResult(self.0, i)))
    }

    pub fn get_single_constant(&self) -> i64 {
        do_unsafe!(mlirAffineMapGetSingleConstantResult(self.0))
    }

    pub fn get_sub_map(&self, pos: &mut [isize]) -> Self {
        Self::from(do_unsafe!(mlirAffineMapGetSubMap(self.0, pos.len() as isize, pos.as_mut_ptr())))
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirAffineMapAttrGetTypeID()))
    }

    pub fn is_empty(&self) -> bool {
        do_unsafe!(mlirAffineMapIsEmpty(self.0))
    }

    pub fn is_identity(&self) -> bool {
        do_unsafe!(mlirAffineMapIsIdentity(self.0))
    }

    pub fn is_minor_identity(&self) -> bool {
        do_unsafe!(mlirAffineMapIsMinorIdentity(self.0))
    }

    pub fn is_permutation(&self) -> bool {
        do_unsafe!(mlirAffineMapIsPermutation(self.0))
    }

    pub fn is_projected_permutation(&self) -> bool {
        do_unsafe!(mlirAffineMapIsProjectedPermutation(self.0))
    }

    pub fn is_single_constant(&self) -> bool {
        do_unsafe!(mlirAffineMapIsSingleConstant(self.0))
    }

    pub fn num_dims(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumDims(self.0))
    }

    /// Equal to `num_dims() + num_symbols()`
    pub fn num_inputs(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumInputs(self.0))
    }

    pub fn num_results(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumResults(self.0))
    }

    pub fn num_symbols(&self) -> isize {
        do_unsafe!(mlirAffineMapGetNumSymbols(self.0))
    }

    pub fn replace_expr(&self, old: &Expr, new: &Expr, num_dims: isize, num_syms: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineMapReplace(self.0, *old.get(), *new.get(), num_dims, num_syms)))
    }
}

impl Symbol {
    pub fn new(context: &Context, i: isize) -> Self {
        Self::from(do_unsafe!(mlirAffineSymbolExprGet(*context.get(), i)))
    }

    pub fn from(expr: MlirAffineExpr) -> Self {
        let expr_ = Expr::from(expr);
        if !expr_.is_symbol() {
            eprint!("Affine expression given cannot be coerced to a symbol: ");
            expr_.dump();
            eprintln!();
            exit(ExitCode::DialectError);
        }
        Symbol(expr)
    }

    pub fn get(&self) -> &MlirAffineExpr {
        &self.0
    }

    pub fn get_position(&self) -> isize {
        do_unsafe!(mlirAffineSymbolExprGetPosition(self.0))
    }
}

impl AffineExpr for Binary {
    fn as_expr(&self) -> Expr {
        Expr::from(self.0)
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl AffineExpr for Constant {
    fn as_expr(&self) -> Expr {
        Expr::from(self.0)
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl AffineExpr for Dim {
    fn as_expr(&self) -> Expr {
        Expr::from(self.0)
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl AffineExpr for Expr {
    fn as_expr(&self) -> Expr {
        self.clone()
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl AffineExpr for Symbol {
    fn as_expr(&self) -> Expr {
        Expr::from(self.0)
    }

    fn get(&self) -> &MlirAffineExpr {
        self.get()
    }
}

impl cmp::PartialEq for dyn AffineExpr {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAffineExprEqual(*self.get(), *rhs.get()))
    }
}

impl From<MlirAffineMap> for Map {
    fn from(attr: MlirAffineMap) -> Self {
        Self(attr)
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
