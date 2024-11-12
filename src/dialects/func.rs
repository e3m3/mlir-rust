// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;
use mlir::MlirOperation;

use std::cmp;
use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::array::Array;
use attributes::dictionary::Dictionary;
use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::named::Named;
use attributes::string::String as StringAttr;
use attributes::symbol_ref::SymbolRef;
use attributes::r#type::Type as TypeAttr;
use dialects::IROp;
use dialects::IROperation;
use dialects::OperandSegmentSizes;
use dialects::ResultSegmentSizes;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Attribute;
use ir::Block;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::OperationState;
use ir::StringBacked;
use ir::StringRef;
use ir::Region;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::function::Function;
use types::IRType;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct Arguments(MlirAttribute);

#[derive(Clone)]
pub struct Callee(MlirAttribute);

#[derive(Clone)]
pub struct FunctionAttr(MlirAttribute);

#[derive(Clone)]
pub struct Referee(MlirAttribute);

#[derive(Clone)]
pub struct Results(MlirAttribute);

#[derive(Clone)]
pub struct SymbolName(MlirAttribute);

#[derive(Clone)]
pub struct SymbolVisibility(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[derive(Clone,Copy,PartialEq)]
pub enum Op {
    Call,
    CallIndirect,
    Constant,
    Func,
    Return,
}

#[derive(Clone,Copy,Default,PartialEq)]
pub enum SymbolVisibilityKind {
    #[default]
    None,
    Private,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Call(MlirOperation);

#[derive(Clone)]
pub struct CallIndirect(MlirOperation);

#[derive(Clone)]
pub struct Constant(MlirOperation);

#[derive(Clone)]
pub struct Func(MlirOperation);

#[derive(Clone)]
pub struct Return(MlirOperation, SymbolRef);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl Arguments {
    pub fn new(context: &Context, elements: &[Dictionary]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::__from(*attr.get())
    }

    fn __from(attr: MlirAttribute) -> Self {
        Arguments(attr)
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let args = Self::__from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes for arguments");
            exit(ExitCode::DialectError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_dictionary()) {
            eprintln!("Expected array of dictionary attributes for arguments");
            exit(ExitCode::DialectError);
        }
        args
    }

    pub fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    pub fn as_dictionaries(&self) -> Vec<Dictionary> {
        let args = self.as_array();
        (0..args.num_elements()).map(|i| Dictionary::from(*args.get_element(i).get())).collect()
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "arg_attrs"
    }
}

impl Callee {
    pub fn new(sym: &SymbolRef) -> Self {
        Self::from(*sym.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let callee = Callee(attr);
        if !callee.as_attribute().is_flat_symbol_ref() {
            eprintln!("Expected flat symbol reference for callee name");
            exit(ExitCode::DialectError);
        }
        callee
    }

    pub fn as_symbol_ref(&self) -> SymbolRef {
        SymbolRef::from(*self.get())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "callee"
    }
}

impl FunctionAttr {
    pub fn new(f: &Function) -> Self {
        let attr = TypeAttr::new(&f.as_type());
        Self::from(*attr.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let f = FunctionAttr(attr);
        let a = f.as_attribute();
        if !a.is_type() || !TypeAttr::from(*a.get()).get_type().is_function() {
            eprintln!("Expected typed function attribute");
            exit(ExitCode::DialectError);
        }
        f
    }

    pub fn as_type_attribute(&self) -> TypeAttr {
        TypeAttr::from(*self.get())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "function_type"
    }

    pub fn get_type(&self) -> Function {
        Function::from(*self.as_type_attribute().get_type().get())
    }
}

impl Referee {
    pub fn new(sym: &SymbolRef) -> Self {
        Self::from(*sym.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let referee = Referee(attr);
        if !referee.as_attribute().is_flat_symbol_ref() {
            eprintln!("Expected flat symbol reference for referee name");
            exit(ExitCode::DialectError);
        }
        referee
    }

    pub fn as_symbol_ref(&self) -> SymbolRef {
        SymbolRef::from(*self.get())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "value"
    }
}

impl Results {
    pub fn new(context: &Context, elements: &[Dictionary]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::__from(*attr.get())
    }

    fn __from(attr: MlirAttribute) -> Self {
        Results(attr)
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let args = Self::__from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes for results");
            exit(ExitCode::DialectError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_dictionary()) {
            eprintln!("Expected array of dictionary attributes for results");
            exit(ExitCode::DialectError);
        }
        args
    }

    pub fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    pub fn as_dictionaries(&self) -> Vec<Dictionary> {
        let args = self.as_array();
        (0..args.num_elements()).map(|i| Dictionary::from(*args.get_element(i).get())).collect()
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "res_attrs"
    }
}

impl SymbolName {
    pub fn new(context: &Context, s: &StringRef) -> Self {
        let sym = StringAttr::new(context, s);
        Self::from(*sym.get())
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let sym = SymbolName(attr);
        if !sym.as_attribute().is_string() {
            eprintln!("Expected string attribute for symbol name");
            exit(ExitCode::DialectError);
        }
        sym
    }

    pub fn as_string(&self) -> StringAttr {
        StringAttr::from(*self.get())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "sym_name"
    }
}

impl SymbolVisibility {
    pub fn new(context: &Context, k: SymbolVisibilityKind) -> Option<Self> {
        match k {
            SymbolVisibilityKind::None      => None,
            SymbolVisibilityKind::Private   => {
                let s = StringBacked::from_string(&k.to_string());
                let sym = StringAttr::new(context, &s.as_string_ref());
                Some(Self::from(*sym.as_attribute().get()))
            },
        }
    }

    pub fn from(attr: MlirAttribute) -> Self {
        let sym = SymbolVisibility(attr);
        if !sym.as_attribute().is_string() {
            eprintln!("Expected string attribute for symbol visibility");
            exit(ExitCode::DialectError);
        }
        sym
    }

    pub fn as_string(&self) -> StringAttr {
        StringAttr::from(*self.get())
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }

    pub fn get_name() -> &'static str {
        "sym_visibility"
    }
}

///////////////////////////////
//  Enum Implementation
///////////////////////////////

impl Op {
    pub fn get_name(&self) -> &'static str {
        match self {
            Op::Call            => "call",
            Op::CallIndirect    => "call_indirect",
            Op::Constant        => "constant",
            Op::Func            => "func",
            Op::Return          => "return",
        }
    }
}

///////////////////////////////
//  Operation Implementation
///////////////////////////////

impl Call {
    pub fn new(callee: &Callee, t: &[Type], args: &[Value], loc: &Location) -> Self {
        let context = callee.get_context();
        let dialect = context.get_dialect_func();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Call.get_name(),
        ));
        let opseg_attr = OperandSegmentSizes::new(&context, &[args.len() as i64]);
        let result_attr = ResultSegmentSizes::new(&context, &[t.len() as i64]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[
            callee.as_named_attribute(),
            opseg_attr.as_named_attribute(),
            result_attr.as_named_attribute(),
        ]);
        op_state.add_operands(args);
        op_state.add_results(t);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Call(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_callee(&self) -> Callee {
        let attr_name = StringBacked::from_string(&Callee::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Callee::from(*attr.get())
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl CallIndirect {
    pub fn new(f: &Value, args: &[Value], loc: &Location) -> Self {
        if !f.is_result() || !f.get_type().is_function() {
            eprintln!("Expected function value type for indirect callee");
            exit(ExitCode::DialectError);
        }
        let t_f = Function::from(*f.get_type().get());
        if t_f.num_inputs() != args.len() as isize {
            eprintln!("Expected number of inputs to match for callee and arguments provided");
            exit(ExitCode::DialectError);
        }
        for (i, arg) in args.iter().enumerate() {
            if t_f.get_input(i as isize) != arg.get_type() {
                eprintln!("Expected matching types for callee type anad argument type at position {}", i);
                exit(ExitCode::DialectError);
            }
        }
        let mut args_: Vec<Value> = vec![f.clone()];
        args_.append(&mut args.to_vec());
        let t: Vec<Type> = (0..t_f.num_results()).map(|i| t_f.get_result(i)).collect();
        let context = f.get_type().get_context();
        let dialect = context.get_dialect_func();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::CallIndirect.get_name(),
        ));
        let opseg_attr = OperandSegmentSizes::new(&context, &[1, args.len() as i64]);
        let result_attr = ResultSegmentSizes::new(&context, &[t.len() as i64]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute(), result_attr.as_named_attribute()]);
        op_state.add_operands(&args_);
        op_state.add_results(&t);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        CallIndirect(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_callee(&self) -> Value {
        self.as_operation().get_operand(0)
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Constant {
    pub fn new(op: &Func, loc: &Location) -> Self {
        let context = op.get_context();
        let dialect = context.get_dialect_func();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Constant.get_name(),
        ));
        let attr = Referee::new(&op.get_symbol_ref());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr.as_named_attribute()]);
        op_state.add_results(&[op.get_type().as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Constant(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_value(&self) -> Referee {
        let attr_name = StringBacked::from_string(&Referee::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Referee::from(*attr.get())
    }
}

impl Func {
    pub fn new(
        t: &Function,
        f_name: &StringRef,
        visibility: SymbolVisibilityKind,
        attr_args: &Arguments,
        attr_results: &Results,
        loc: &Location,
    ) -> Self {
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_func();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Func.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        let mut block = Block::default();
        let _operands: Vec<Value> = (0..t.num_inputs())
            .map(|i| block.add_arg(&t.get_input(i), loc))
            .collect();
        let sym_name_attr = SymbolName::new(&context, f_name);
        let function_type_attr = FunctionAttr::new(t);
        let mut attrs: Vec<Named> = Vec::new();
        attrs.push(sym_name_attr.as_named_attribute());
        attrs.push(function_type_attr.as_named_attribute());
        match SymbolVisibility::new(&context, visibility) {
            Some(sym)   => attrs.push(sym.as_named_attribute()),
            None        => {
                let mut region = Region::new();
                region.append_block(&mut block);
                op_state.add_regions(&[region]);
            },
        };
        attrs.push(attr_args.as_named_attribute());
        attrs.push(attr_results.as_named_attribute());
        op_state.add_attributes(&attrs);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Func(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_arguments(&self) -> Arguments {
        let attr_name = StringBacked::from_string(&Arguments::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Arguments::from(*attr.get())
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_function(&self) -> FunctionAttr {
        let attr_name = StringBacked::from_string(&FunctionAttr::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        FunctionAttr::from(*attr.get())
    }

    pub fn get_symbol_name(&self) -> SymbolName {
        let attr_name = StringBacked::from_string(&SymbolName::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        SymbolName::from(*attr.get())
    }

    pub fn get_symbol_ref(&self) -> SymbolRef {
        SymbolRef::new_flat(&self.get_context(), &self.get_symbol_name().as_string().get_string())
    }

    pub fn get_result_attributes(&self) -> Results {
        let attr_name = StringBacked::from_string(&Results::get_name().to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Results::from(*attr.get())
    }

    pub fn get_type(&self) -> Function {
        self.get_function().get_type()
    }

    pub fn get_visibility(&self) -> Option<SymbolVisibility> {
        let op = self.as_operation();
        let attr_name = StringBacked::from_string(&SymbolVisibility::get_name().to_string());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(SymbolVisibility::from(*attr.get()))
        } else {
            None
        }
    }
}

impl Return {
    pub fn new(parent: &Func, args: &[Value], loc: &Location) -> Self {
        let t_f = parent.get_type();
        let num_results = t_f.num_results() as usize;
        let symbol_ref = parent.get_symbol_ref();
        if num_results != args.len() {
            eprintln!("Expected '{}' results for func.func '@{}'", num_results, symbol_ref);
            exit(ExitCode::DialectError);
        }
        for i in 0..num_results {
            let t = t_f.get_result(i as isize);
            if t != args.get(i).unwrap().get_type() {
                eprintln!("Expected matching type for func.func '@{}' result at position '{}'",
                    symbol_ref,
                    i,
                );
                exit(ExitCode::DialectError);
            }
        }
        let context = parent.get_context();
        let dialect = context.get_dialect_func();
        let name = StringBacked::from_string(&format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Return.get_name(),
        ));
        let opseg_attr = OperandSegmentSizes::new(&context, &[args.len() as i64]);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[opseg_attr.as_named_attribute()]);
        op_state.add_operands(args);
        Self::from(*op_state.create_operation().get(), symbol_ref)
    }

    pub fn from(op: MlirOperation, parent: SymbolRef) -> Self {
        Return(op, parent)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_parent(&self) -> &SymbolRef {
        &self.1
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl IRAttribute for Arguments {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for Arguments {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IROperation for Call {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_func()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CallOpInterface,
            Interface::SymbolUserOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Call.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Call
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::MemRefsNormalizable,
        ]
    }
}

impl IRAttribute for Callee {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for Callee {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IROperation for CallIndirect {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_func()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CallOpInterface
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::CallIndirect.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::CallIndirect
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl IROperation for Constant {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_func()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::SymbolUserOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Constant.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Constant
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ConstantLike,
        ]
    }
}

impl IROperation for Func {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_func()
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CallableOpInterface,
            Interface::FunctionOpInterface,
            Interface::OpAsmOpInterface,
            Interface::Symbol,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Func.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Func
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AffineScope,
            Trait::AutomaticAllocationScope,
            Trait::IsolatedFromAbove,
        ]
    }
}

impl IRAttribute for FunctionAttr {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for FunctionAttr {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IROp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl IRAttribute for Referee {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for Referee {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IRAttribute for Results {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for Results {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IROperation for Return {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_func()
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
        Op::Return.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Return
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::HasParent(&[&Op::Func]),
            Trait::MemRefsNormalizable,
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

impl cmp::PartialEq for Return {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation() && self.get_parent() == rhs.get_parent()
    }
}

impl IRAttribute for SymbolName {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for SymbolName {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl IRAttribute for SymbolVisibility {
    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IRAttributeNamed for SymbolVisibility {
    fn get_name(&self) -> &'static str {
        Self::get_name()
    }
}

impl FromStr for SymbolVisibilityKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            ""          => Ok(SymbolVisibilityKind::None),
            "private"   => Ok(SymbolVisibilityKind::Private),
            _           => Err(format!("Invalid symbol visibility kind: {}", s)),
        }
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::CallIndirect    => "CallIndirectOp",
            Op::Call            => "CallOp",
            Op::Constant        => "ConstantOp",
            Op::Func            => "FuncOp",
            Op::Return          => "ReturnOp",
        })
    }
}

impl fmt::Display for SymbolVisibilityKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            SymbolVisibilityKind::None      => "none",
            SymbolVisibilityKind::Private   => "private",
        })
    }
}
