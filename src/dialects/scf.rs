// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirOperation;

use std::fmt;
use std::iter;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use dialects::IOp;
use dialects::IOperation;
use dialects::OpRef;
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
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::IType;

///////////////////////////////
//  Attributes
///////////////////////////////

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Op {
    Condition,
    ExecuteRegion,
    For,
    Forall,
    ForallInParallel,
    If,
    IndexSwitch,
    Parallel,
    Reduce,
    ReduceReturn,
    While,
    Yield,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Condition(MlirOperation, MlirOperation);

#[derive(Clone)]
pub struct ExecuteRegion(MlirOperation);

#[derive(Clone)]
pub struct For(MlirOperation);

#[derive(Clone)]
pub struct Forall(MlirOperation);

#[derive(Clone)]
pub struct ForallInParallel(MlirOperation);

#[derive(Clone)]
pub struct If(MlirOperation);

#[derive(Clone)]
pub struct IndexSwitch(MlirOperation);

#[derive(Clone)]
pub struct Parallel(MlirOperation);

#[derive(Clone)]
pub struct Reduce(MlirOperation);

#[derive(Clone)]
pub struct ReduceReturn(MlirOperation);

#[derive(Clone)]
pub struct While(MlirOperation);

#[derive(Clone)]
pub struct Yield(MlirOperation, MlirOperation, Op);

///////////////////////////////
//  Attribute Implementations
///////////////////////////////

///////////////////////////////
//  Enum Implementations
///////////////////////////////

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::Condition => "condition",
            Op::ExecuteRegion => "execute_region",
            Op::For => "for",
            Op::Forall => "forall",
            Op::ForallInParallel => "forall.in_parallel",
            Op::If => "if",
            Op::IndexSwitch => "index_switch",
            Op::Parallel => "parallel",
            Op::Reduce => "reduce",
            Op::ReduceReturn => "reduce.return",
            Op::While => "while",
            Op::Yield => "yield",
        }
    }
}

///////////////////////////////
//  Operation Implementations
///////////////////////////////

impl Condition {
    pub fn new(
        context: &Context,
        parent: &While,
        cond: &Value,
        args: &[Value],
        loc: &Location,
    ) -> Self {
        let n_results = parent.as_operation().num_results();
        let n_args = args.len() as isize;
        if n_results != n_args {
            eprintln!(
                "Expected matching number of parent results ({}) and operands ({}) of condition operation",
                n_results, n_args,
            );
            exit(ExitCode::DialectError);
        }
        if !cond.get_type().is_bool() {
            eprintln!("Expected bool type for condition operand of condition operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::Condition);
        let mut operands = vec![cond.clone()];
        operands.append(&mut args.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&operands);
        Self::from((*op_state.create_operation().get(), *parent.get()))
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
        &Op::While
    }

    pub fn get_parent_mut(&mut self) -> &mut MlirOperation {
        &mut self.1
    }
}

impl ExecuteRegion {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl For {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Forall {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl ForallInParallel {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl If {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl IndexSwitch {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Parallel {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Reduce {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl ReduceReturn {
    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl While {
    pub fn new(context: &Context, results: &[Type], inits: &[Value], loc: &Location) -> Self {
        let n_results = results.len();
        let n_inits = inits.len();
        if n_results != n_inits {
            eprintln!(
                "Expected matching number of results ({}) and init operands ({}) for while operation",
                n_results, n_inits,
            );
            exit(ExitCode::DialectError);
        }
        if iter::zip(results.iter(), inits.iter()).any(|(r, v)| *r != v.get_type()) {
            eprintln!("Expected matching result and init operand types for while operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::While);
        let locs: Vec<Location> = inits.iter().map(|_| loc.clone()).collect();
        let t_inits: Vec<Type> = inits.iter().map(|v| v.get_type()).collect();
        let mut region_before = Region::new();
        let mut region_after = Region::new();
        let mut block_before = Block::new(inits.len() as isize, &t_inits, &locs);
        let mut block_after = Block::new(inits.len() as isize, &t_inits, &locs);
        region_before.append_block(&mut block_before);
        region_after.append_block(&mut block_after);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_regions(&[region_before, region_after]);
        if !inits.is_empty() {
            op_state.add_operands(inits);
        }
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

    pub fn get_region_after(&self) -> Region {
        self.as_operation().get_region(1)
    }

    pub fn get_region_before(&self) -> Region {
        self.as_operation().get_region(0)
    }
}

impl Yield {
    fn new(
        context: &Context,
        results: &[Type],
        parent: &dyn IOperation,
        parent_op: Op,
        values: &[Value],
        loc: &Location,
    ) -> Self {
        let n_results = results.len() as isize;
        let n_results_parent = parent.as_operation().num_results();
        let n_values = values.len() as isize;
        if n_results != n_results_parent {
            eprintln!(
                "Expected matching number of results ({}) and parent results ({}) for yield operation",
                n_results, n_results_parent,
            );
            exit(ExitCode::DialectError);
        }
        if n_results != n_values {
            eprintln!(
                "Expected matching number of results ({}) and value operands ({}) for yield operation",
                n_results, n_values,
            );
            exit(ExitCode::DialectError);
        }
        if iter::zip(results.iter(), values.iter()).any(|(r, v)| *r != v.get_type()) {
            eprintln!("Expected matching result and value operand types for yield operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::Yield);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        if !values.is_empty() {
            op_state.add_operands(values);
        }
        Self::from((*op_state.create_operation().get(), *parent.get(), parent_op))
    }

    pub fn new_execute_region(
        results: &[Type],
        parent: &ExecuteRegion,
        values: &[Value],
        loc: &Location,
    ) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            results,
            parent,
            Op::ExecuteRegion,
            values,
            loc,
        )
    }

    pub fn new_for(results: &[Type], parent: &For, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            results,
            parent,
            Op::For,
            values,
            loc,
        )
    }

    pub fn new_if(results: &[Type], parent: &If, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            results,
            parent,
            Op::If,
            values,
            loc,
        )
    }

    pub fn new_index_switch(
        results: &[Type],
        parent: &IndexSwitch,
        values: &[Value],
        loc: &Location,
    ) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            results,
            parent,
            Op::IndexSwitch,
            values,
            loc,
        )
    }

    pub fn new_while(results: &[Type], parent: &While, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            results,
            parent,
            Op::While,
            values,
            loc,
        )
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

    pub fn get_parent_mut(&mut self) -> &mut MlirOperation {
        &mut self.1
    }

    pub fn get_parent_mut_op(&mut self) -> &mut Op {
        &mut self.2
    }
}

///////////////////////////////
//  Trait Implementations
///////////////////////////////

impl From<(MlirOperation, MlirOperation)> for Condition {
    fn from((op, parent): (MlirOperation, MlirOperation)) -> Self {
        Self(op, parent)
    }
}

impl IOperation for Condition {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
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
        Op::Condition.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Condition
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::HasParent(&[&Op::While]),
            Trait::Terminator,
        ]
    }
}

impl From<MlirOperation> for ExecuteRegion {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ExecuteRegion {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::RegionBranchOpInterface]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ExecuteRegion.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::ExecuteRegion
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
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
        self.as_operation().get_context().get_dialect_scf()
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
            Trait::AutomaticAllocationScope,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

impl From<MlirOperation> for Forall {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Forall {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::DestinationStyleOpInterface,
            Interface::LoopLikeOpInterface,
            Interface::RegionBranchOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Forall.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Forall
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AttrSizedOperandSegments,
            Trait::AutomaticAllocationScope,
            Trait::HasParallelRegion,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::ForallInParallel]),
            Trait::SingleBlock,
        ]
    }
}

impl From<MlirOperation> for ForallInParallel {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ForallInParallel {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::ParallelCombiningOpInterface,
            Interface::RegionKindInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::ForallInParallel.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::ForallInParallel
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::HasOnlyGraphRegion,
            Trait::HasParent(&[&Op::Forall]),
            Trait::NoTerminator,
            Trait::SingleBlock,
            Trait::Terminator,
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
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::InferTypeOpInterface,
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
            Trait::InferTypeOpAdaptor,
            Trait::NoRegionArguments,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

impl From<MlirOperation> for IndexSwitch {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for IndexSwitch {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::RegionBranchOpInterface]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::IndexSwitch.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::IndexSwitch
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::Yield]),
            Trait::SingleBlock,
        ]
    }
}

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
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::LoopLikeOpInterface,
            Interface::RegionBranchOpInterface,
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
            Trait::AttrSizedOperandSegments,
            Trait::AutomaticAllocationScope,
            Trait::HasParallelRegion,
            Trait::RecursiveMemoryEffects,
            Trait::SingleBlockImplicitTerminator(&[&Op::Reduce]),
            Trait::SingleBlock,
        ]
    }
}

impl From<MlirOperation> for Reduce {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Reduce {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::RegionBranchOpInterface]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Reduce.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Reduce
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::HasParent(&[&Op::Parallel]),
            Trait::RecursiveMemoryEffects,
            Trait::Terminator,
        ]
    }
}

impl From<MlirOperation> for ReduceReturn {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for ReduceReturn {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
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
        Op::Reduce.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Reduce
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::HasParent(&[&Op::Reduce]),
            Trait::Terminator,
        ]
    }
}

impl From<MlirOperation> for While {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for While {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::LoopLikeOpInterface,
            Interface::RegionBranchOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::While.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::While
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::RecursiveMemoryEffects, Trait::SingleBlock]
    }
}

impl From<(MlirOperation, MlirOperation, Op)> for Yield {
    fn from((op, parent, parent_op): (MlirOperation, MlirOperation, Op)) -> Self {
        Self(op, parent, parent_op)
    }
}

impl IOperation for Yield {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_scf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::RegionBranchOpInterface,
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
            Trait::HasParent(&[
                &Op::ExecuteRegion,
                &Op::For,
                &Op::If,
                &Op::IndexSwitch,
                &Op::While,
            ]),
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Condition => "ConditionOp",
            Op::ExecuteRegion => "ExecuteRegionOp",
            Op::For => "ForOp",
            Op::Forall => "ForallOp",
            Op::ForallInParallel => "InParallelOP",
            Op::If => "IfOp",
            Op::IndexSwitch => "IndexSwitchOp",
            Op::Parallel => "ParallelOp",
            Op::Reduce => "ReduceOp",
            Op::ReduceReturn => "ReduceReturnOp",
            Op::While => "WhileOp",
            Op::Yield => "YieldOp",
        })
    }
}
