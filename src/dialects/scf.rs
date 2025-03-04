// Copyright 2025, Giordano Salvador
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

use attributes::IAttributeNamed;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedI64DenseArray;
use attributes::specialized::SpecializedAttribute;
use dialects::IOp;
use dialects::IOperation;
use dialects::OpRef;
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
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::IType;
use types::index::Index;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct Cases(MlirAttribute);

#[derive(Clone)]
pub struct Mapping(MlirAttribute);

#[derive(Clone)]
pub struct StaticLowerBound(MlirAttribute);

#[derive(Clone)]
pub struct StaticUpperBound(MlirAttribute);

#[derive(Clone)]
pub struct StaticStep(MlirAttribute);

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
pub struct Reduce(MlirOperation, MlirOperation);

#[derive(Clone)]
pub struct ReduceReturn(MlirOperation, MlirOperation);

#[derive(Clone)]
pub struct While(MlirOperation);

#[derive(Clone)]
pub struct Yield(MlirOperation, MlirOperation, Op);

///////////////////////////////
//  Attribute Implementations
///////////////////////////////

impl Cases {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Mapping {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticLowerBound {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticUpperBound {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl StaticStep {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

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
    pub fn new(context: &Context, results: &[Type], loc: &Location) -> Self {
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::ExecuteRegion);
        let region = Region::new();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_regions(&[region]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region(&self) -> Region {
        self.as_operation().get_region(0)
    }
}

impl For {
    pub fn new(
        context: &Context,
        results: &[Type],
        lower_bound: &Value,
        upper_bound: &Value,
        step: &Value,
        inits: &[Value],
        loc: &Location,
    ) -> Self {
        let t_lower = lower_bound.get_type();
        let t_upper = upper_bound.get_type();
        let t_step = step.get_type();
        if !t_lower.is_index() && !t_lower.is_integer() {
            eprintln!(
                "Expected index or integer type for bounds and step operands of for operation"
            );
            exit(ExitCode::DialectError);
        }
        if t_lower != t_upper || t_lower != t_step {
            eprintln!("Expected bounds and step types to all be the same type for for operation");
            exit(ExitCode::DialectError);
        }
        let n_results = results.len();
        let n_inits = inits.len();
        if n_results != n_inits {
            eprintln!(
                "Expected matching number of results ({}) and init ({}) operands of for operation",
                n_results, n_inits,
            );
            exit(ExitCode::DialectError);
        } else if iter::zip(results.iter(), inits.iter()).any(|(r, v)| *r != v.get_type()) {
            eprintln!("Expected matching type for results and init operands of for operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::For);
        let mut t_block: Vec<Type> = vec![step.get_type()];
        let mut locs: Vec<Location> = vec![loc.clone()];
        inits.iter().for_each(|v| t_block.push(v.get_type()));
        inits.iter().for_each(|_| locs.push(loc.clone()));
        let mut region = Region::new();
        let mut block = Block::new(t_block.len() as isize, &t_block, &locs);
        region.append_block(&mut block);
        let mut operands = vec![lower_bound.clone(), upper_bound.clone(), step.clone()];
        operands.append(&mut inits.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&operands);
        op_state.add_regions(&[region]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region(&self) -> Region {
        self.as_operation().get_region(0)
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
    fn new(
        op_state: &mut OperationState,
        region_then: Region,
        region_else: Region,
        results: &[Type],
        cond: &Value,
    ) -> Self {
        Self::check_operands(results, cond);
        op_state.add_operands(&[cond.clone()]);
        op_state.add_regions(&[region_then, region_else]);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn new_if(context: &Context, results: &[Type], cond: &Value, loc: &Location) -> Self {
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::If);
        let mut region_then = Region::new();
        let region_else = Region::new();
        let mut block_then = Block::new_empty();
        region_then.append_block(&mut block_then);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        Self::new(&mut op_state, region_then, region_else, results, cond)
    }

    pub fn new_if_else(context: &Context, results: &[Type], cond: &Value, loc: &Location) -> Self {
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::If);
        let mut region_then = Region::new();
        let mut region_else = Region::new();
        let mut block_then = Block::new_empty();
        let mut block_else = Block::new_empty();
        region_then.append_block(&mut block_then);
        region_else.append_block(&mut block_else);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        Self::new(&mut op_state, region_then, region_else, results, cond)
    }

    fn check_operands(_results: &[Type], cond: &Value) -> () {
        if !cond.get_type().is_bool() {
            eprintln!("Expected bool type for condition operand of if operation");
            exit(ExitCode::DialectError);
        }
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region_else(&self) -> Region {
        self.as_operation().get_region(1)
    }

    pub fn get_region_then(&self) -> Region {
        self.as_operation().get_region(0)
    }
}

impl IndexSwitch {
    pub fn new(
        context: &Context,
        results: &[Type],
        value: &Value,
        cases: &[i64],
        loc: &Location,
    ) -> Self {
        if !value.get_type().is_index() {
            eprintln!("Expected index type for value operand of index switch operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::IndexSwitch);
        let attr = Cases::new(context, cases).as_named_attribute();
        let mut region_default = Region::new();
        let mut block_default = Block::new_empty();
        region_default.append_block(&mut block_default);
        let mut regions: Vec<Region> = vec![region_default];
        for _ in cases.iter() {
            let mut region = Region::new();
            let mut block = Block::new_empty();
            region.append_block(&mut block);
            regions.push(region);
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr]);
        op_state.add_operands(&[value.clone()]);
        op_state.add_regions(&regions);
        if !results.is_empty() {
            op_state.add_results(results);
        }
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_cases(&self) -> Cases {
        let attr_name = StringBacked::from(Cases::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Cases::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region(&self, i: isize) -> Region {
        self.as_operation().get_region(i)
    }

    pub fn get_region_default(&self) -> Region {
        self.as_operation().get_region(0)
    }

    /// Number of cases in the IndexSwitch operation, excluding the default case.
    pub fn num_cases(&self) -> isize {
        self.as_operation().num_regions() - 1
    }

    pub fn num_regions(&self) -> isize {
        self.as_operation().num_regions()
    }
}

impl Parallel {
    pub fn new(
        context: &Context,
        lower_bound: &Value,
        upper_bound: &Value,
        step: &Value,
        inits: &[Value],
        loc: &Location,
    ) -> Self {
        if !lower_bound.get_type().is_index() {
            eprintln!("Expected index type for lower bound operand of parallel operation");
            exit(ExitCode::DialectError);
        }
        if !upper_bound.get_type().is_index() {
            eprintln!("Expected index type for upper bound operand of parallel operation");
            exit(ExitCode::DialectError);
        }
        if !step.get_type().is_index() {
            eprintln!("Expected index type for step operand of parallel operation");
            exit(ExitCode::DialectError);
        }
        let t_index = Index::new(context).as_type();
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::Parallel);
        let mut operands = vec![lower_bound.clone(), upper_bound.clone(), step.clone()];
        let mut region = Region::new();
        let mut block = Block::new(1, &[t_index], &[loc.clone()]);
        region.append_block(&mut block);
        let attr_opseg =
            OperandSegmentSizes::new(context, &[1, 1, 1, inits.len() as i32]).as_named_attribute();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        if !inits.is_empty() {
            let results: Vec<Type> = inits.iter().map(|v| v.get_type()).collect();
            op_state.add_results(&results);
            operands.append(&mut inits.to_vec());
        }
        op_state.add_attributes(&[attr_opseg]);
        op_state.add_operands(&operands);
        op_state.add_regions(&[region]);
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_region(&self) -> Region {
        self.as_operation().get_region(0)
    }
}

impl Reduce {
    /// Number of reductions (`num_reductions`) determines how many regions are instantiated for
    /// the operation.
    pub fn new(
        context: &Context,
        parent: &Parallel,
        args: &[Value],
        num_reductions: usize,
        loc: &Location,
    ) -> Self {
        if args.is_empty() && num_reductions > 0 {
            eprintln!("Expected no reduction regions for empty reduce operation");
            exit(ExitCode::DialectError);
        }
        let op_parent = parent.as_operation();
        let n_results = op_parent.num_results();
        let n_args = args.len() as isize;
        if n_results != n_args {
            eprintln!(
                "Expected matching number of parent operation results ({}) and arguments ({}) \
                of reduce operation",
                n_results, n_args,
            );
            exit(ExitCode::DialectError);
        } else if n_results > 0 {
            for (i, arg) in args.iter().enumerate() {
                let t = op_parent.get_result(i as isize).get_type();
                if t != arg.get_type() {
                    eprintln!(
                        "Expected matching type of parent operation result and argument {} \
                        of reduce operation",
                        i
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::Reduce);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        match args {
            [] => (),
            [arg, ..] => {
                let t = arg.get_type();
                let mut regions: Vec<Region> = vec![];
                for _ in 0..num_reductions {
                    let mut region = Region::new();
                    let mut block =
                        Block::new(2, &[t.clone(), t.clone()], &[loc.clone(), loc.clone()]);
                    region.append_block(&mut block);
                    regions.push(region);
                }
                op_state.add_operands(args);
                op_state.add_regions(&regions);
            }
        }
        Self::from((*op_state.create_operation().get_mut(), *parent.get()))
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

    pub fn get_parent_mut(&mut self) -> &mut MlirOperation {
        &mut self.1
    }

    pub fn get_region(&self, i: isize) -> Option<Region> {
        let op = self.as_operation();
        if op.num_regions() > 0 {
            Some(op.get_region(i))
        } else {
            None
        }
    }
}

impl ReduceReturn {
    /// Uses the pos-th region in the parent to check for expected result types.
    /// Each region in the parent is expected to have the same block argument signature.
    pub fn new(
        context: &Context,
        parent: &Reduce,
        value: &Value,
        pos: usize,
        loc: &Location,
    ) -> Self {
        let Some(parent_region) = parent.get_region(pos as isize) else {
            eprintln!(
                "Expected region {} in parent reduce operation of reduce return operation",
                pos
            );
            exit(ExitCode::DialectError);
        };
        let parent_block = parent_region.iter().next().unwrap_or_default();
        if parent_block.get_arg(0).get_type() != value.get_type() {
            eprintln!(
                "Expected matching type for reduce operation arguments and value operand \
                of reduce return operation",
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_scf();
        let name = dialect.get_op_name(&Op::ReduceReturn);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        Self::from((*op_state.create_operation().get_mut(), *parent.get()))
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

    pub fn get_parent_mut(&mut self) -> &mut MlirOperation {
        &mut self.1
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
        } else if iter::zip(results.iter(), inits.iter()).any(|(r, v)| *r != v.get_type()) {
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
        parent: &dyn IOperation,
        parent_op: Op,
        values: &[Value],
        loc: &Location,
    ) -> Self {
        let results: Vec<Type> = values.iter().map(|v| v.get_type()).collect();
        let n_results = results.len() as isize;
        let n_results_parent = parent.as_operation().num_results();
        if n_results != n_results_parent {
            eprintln!(
                "Expected matching number of results ({}) and parent results ({}) for yield operation",
                n_results, n_results_parent,
            );
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

    pub fn new_execute_region(parent: &ExecuteRegion, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            parent,
            Op::ExecuteRegion,
            values,
            loc,
        )
    }

    pub fn new_for(parent: &For, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            parent,
            Op::For,
            values,
            loc,
        )
    }

    pub fn new_if(parent: &If, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            parent,
            Op::If,
            values,
            loc,
        )
    }

    pub fn new_index_switch(parent: &IndexSwitch, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
            parent,
            Op::IndexSwitch,
            values,
            loc,
        )
    }

    pub fn new_while(parent: &While, values: &[Value], loc: &Location) -> Self {
        Self::new(
            &parent.as_operation().get_context(),
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

SpecializedAttribute!("cases" = impl NamedI64DenseArray for Cases {});

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
        Self(op, *Operation::new_null().get_mut())
    }
}

impl From<(MlirOperation, MlirOperation)> for Reduce {
    fn from((op, parent_op): (MlirOperation, MlirOperation)) -> Self {
        Self(op, parent_op)
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
        Self(op, *Operation::new_null().get_mut())
    }
}

impl From<(MlirOperation, MlirOperation)> for ReduceReturn {
    fn from((op, parent_op): (MlirOperation, MlirOperation)) -> Self {
        Self(op, parent_op)
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

SpecializedAttribute!("staticLowerBound" = impl NamedI64DenseArray for StaticLowerBound {});

SpecializedAttribute!("staticUpperBound" = impl NamedI64DenseArray for StaticUpperBound {});

SpecializedAttribute!("staticStep" = impl NamedI64DenseArray for StaticStep {});

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
