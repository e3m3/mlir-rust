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
use attributes::specialized::NamedIntegerDenseElements;
use attributes::specialized::NamedString;
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
use ir::OperationState;
use ir::ShapeImpl;
use ir::StringBacked;
use ir::StringRef;
use ir::Value;
use traits::Trait;
use types::IType;
use types::integer::Integer as IntegerType;
use types::vector::Vector;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct CaseValues(MlirAttribute);

#[derive(Clone)]
pub struct CaseOperandSegments(MlirAttribute);

#[derive(Clone)]
pub struct Message(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Op {
    Assert,
    Br,
    CondBr,
    Switch,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Assert(MlirOperation);

#[derive(Clone)]
pub struct Branch(MlirOperation);

#[derive(Clone)]
pub struct CondBranch(MlirOperation);

#[derive(Clone)]
pub struct Switch(MlirOperation);

///////////////////////////////
//  Attribute Implementations
///////////////////////////////

impl CaseValues {
    pub fn new(t: &IntegerType, elements: &[i64]) -> Self {
        let s = ShapeImpl::from(vec![elements.len() as i64]);
        let t_vec = Vector::new(&s, &t.as_type()).as_shaped();
        <Self as NamedIntegerDenseElements>::new(&t_vec, elements)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl CaseOperandSegments {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Message {
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
            Op::Assert => "assert",
            Op::Br => "br",
            Op::CondBr => "cond_br",
            Op::Switch => "switch",
        }
    }
}

///////////////////////////////
//  Operation Implementations
///////////////////////////////

impl Assert {
    pub fn new(context: &Context, cond: &Value, msg: &StringRef, loc: &Location) -> Self {
        if !cond.get_type().is_bool() {
            eprintln!(
                "Expected bool type (1-bit signless integer) for condition operand of \
                assert operation"
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_cf();
        let name = dialect.get_op_name(&Op::Assert);
        let attr = Message::new(context, msg).as_named_attribute();
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr]);
        op_state.add_operands(&[cond.clone()]);
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_message(&self) -> Message {
        let attr_name = StringBacked::from(Message::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Message::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Branch {
    pub fn new(context: &Context, args: &[Value], succ: &Block, loc: &Location) -> Self {
        let n_args = args.len() as isize;
        let n_succ = succ.num_args();
        if n_succ != n_args {
            eprintln!(
                "Expected matching number of successor block arguments ({}) and \
                operands ({}) of branch operation",
                n_succ, n_args,
            );
            exit(ExitCode::DialectError);
        }
        if args
            .iter()
            .enumerate()
            .any(|(i, v)| v.get_type() != succ.get_arg(i as isize).get_type())
        {
            eprintln!(
                "Expected matching types for successor block argument and operands of \
                branch operation"
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_cf();
        let name = dialect.get_op_name(&Op::Br);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(args);
        op_state.add_successors(&[succ.clone()]);
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_successor(&self) -> Block {
        self.as_operation().get_successor(0)
    }
}

impl CondBranch {
    pub fn new(
        context: &Context,
        cond: &Value,
        args_true: &[Value],
        args_false: &[Value],
        succ_true: &Block,
        succ_false: &Block,
        loc: &Location,
    ) -> Self {
        if !cond.get_type().is_bool() {
            eprintln!(
                "Expected bool type (1-bit signless integer) for condition operand of \
                cond branch operation"
            );
            exit(ExitCode::DialectError);
        }
        let n_args_true = args_true.len() as isize;
        let n_args_false = args_false.len() as isize;
        let n_succ_true = succ_true.num_args();
        let n_succ_false = succ_false.num_args();
        if n_succ_true != n_args_true {
            eprintln!(
                "Expected matching number of true successor block arguments ({}) and \
                operands ({}) of cond branch operation",
                n_succ_true, n_args_true,
            );
            exit(ExitCode::DialectError);
        }
        if n_succ_false != n_args_false {
            eprintln!(
                "Expected matching number of false successor block arguments ({}) and \
                operands ({}) of cond branch operation",
                n_succ_true, n_args_true,
            );
            exit(ExitCode::DialectError);
        }
        if args_true
            .iter()
            .enumerate()
            .any(|(i, v)| v.get_type() != succ_true.get_arg(i as isize).get_type())
        {
            eprintln!(
                "Expected matching types for true successor block arguments and operands of \
                cond branch operation"
            );
            exit(ExitCode::DialectError);
        }
        if args_false
            .iter()
            .enumerate()
            .any(|(i, v)| v.get_type() != succ_false.get_arg(i as isize).get_type())
        {
            eprintln!(
                "Expected matching types for false successor block arguments and operands of \
                cond branch operation"
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_cf();
        let name = dialect.get_op_name(&Op::CondBr);
        let attr_opseg = OperandSegmentSizes::new(context, &[
            1,
            args_true.len() as i32,
            args_false.len() as i32,
        ])
        .as_named_attribute();
        let mut operands: Vec<Value> = vec![cond.clone()];
        operands.append(&mut args_true.to_vec());
        operands.append(&mut args_false.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_opseg]);
        op_state.add_operands(&operands);
        op_state.add_successors(&[succ_true.clone(), succ_false.clone()]);
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_successor_false(&self) -> Block {
        self.as_operation().get_successor(1)
    }

    pub fn get_successor_true(&self) -> Block {
        self.as_operation().get_successor(0)
    }
}

impl Switch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &Context,
        flag: &Value,
        cases: &[i64],
        args_default: &[Value],
        args_cases: &[&[Value]],
        succ_default: &Block,
        succ_cases: &[Block],
        loc: &Location,
    ) -> Self {
        let t = flag.get_type();
        if !t.is_integer() {
            eprintln!("Expected integer type for flag operand of switch operation");
            exit(ExitCode::DialectError);
        }
        let case_opseg: Vec<i32> = args_cases.iter().map(|args| args.len() as i32).collect();
        let n_block_args: Vec<i32> = succ_cases.iter().map(|b| b.num_args() as i32).collect();
        if iter::zip(case_opseg.iter(), n_block_args.iter()).any(|(n_c, n_b)| n_c != n_b) {
            eprintln!(
                "Expected matching number of successor block arguments and case arguments \
                for switch operation"
            );
            exit(ExitCode::DialectError);
        }
        for (j, arg) in args_default.iter().enumerate() {
            if arg.get_type() != succ_default.get_arg(j as isize).get_type() {
                eprintln!(
                    "Expected matching types for operands and block argument {} for \
                    default case of switch operation",
                    j,
                );
                exit(ExitCode::DialectError);
            }
        }
        for (i, (args, block)) in iter::zip(args_cases.iter(), succ_cases.iter()).enumerate() {
            for (j, arg) in args.iter().enumerate() {
                if arg.get_type() != block.get_arg(j as isize).get_type() {
                    eprintln!(
                        "Expected matching types for operands and block argument {} for \
                        case {} of switch operation",
                        j, i,
                    );
                    exit(ExitCode::DialectError);
                }
            }
        }
        let dialect = context.get_dialect_cf();
        let name = dialect.get_op_name(&Op::Switch);
        let n_args_cases = case_opseg.iter().sum();
        let attr_opseg =
            OperandSegmentSizes::new(context, &[1, args_default.len() as i32, n_args_cases])
                .as_named_attribute();
        let attr_case_opseg = CaseOperandSegments::new(context, &case_opseg).as_named_attribute();
        let attr_case_values = CaseValues::new(&IntegerType::from(t), cases).as_named_attribute();
        let mut operands: Vec<Value> = vec![flag.clone()];
        operands.append(&mut args_default.to_vec());
        args_cases
            .iter()
            .for_each(|args| operands.append(&mut args.to_vec()));
        let mut successors: Vec<Block> = vec![succ_default.clone()];
        successors.append(&mut succ_cases.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_opseg, attr_case_opseg, attr_case_values]);
        op_state.add_operands(&operands);
        op_state.add_successors(&successors);
        Self::from(*op_state.create_operation().get_mut())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_case_operand_segments(&self) -> CaseOperandSegments {
        let attr_name = StringBacked::from(CaseOperandSegments::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        CaseOperandSegments::from(*attr.get())
    }

    pub fn get_case_values(&self) -> CaseValues {
        let attr_name = StringBacked::from(CaseValues::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        CaseValues::from(*attr.get())
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl From<MlirOperation> for Assert {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Assert {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_cf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::MemoryEffect(
            MemoryEffectOpInterface::UndefinedMemoryEffect,
        )]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Assert.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Assert
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirOperation> for Branch {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Branch {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_cf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::BranchOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Br.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Br
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait, Trait::Terminator]
    }
}

SpecializedAttribute!("case_values" = impl NamedIntegerDenseElements for CaseValues {});

SpecializedAttribute!("case_operand_segments" = impl NamedI32DenseArray for CaseOperandSegments {});

impl From<MlirOperation> for CondBranch {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for CondBranch {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_cf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::BranchOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CondBr.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::CondBr
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::AttrSizedOperandSegments,
            Trait::Terminator,
        ]
    }
}

SpecializedAttribute!("msg" = impl NamedString for Message {});

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for Switch {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IOperation for Switch {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_cf()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::BranchOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Switch.get_name()
    }

    fn get_op(&self) -> OpRef {
        &Op::Switch
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::AttrSizedOperandSegments,
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
            Op::Assert => "AssertOp",
            Op::Br => "BranchOp",
            Op::CondBr => "CondBranchOp",
            Op::Switch => "SwitchOp",
        })
    }
}
