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
use attributes::IRAttribute;
use attributes::named::Named;
use attributes::string::String as StringAttr;
use attributes::symbol_ref::SymbolRef;
use attributes::r#type::Type as TypeAttr;
use dialects::IROp;
use dialects::IROperation;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Attribute;
use ir::Block;
use ir::Context;
use ir::Dialect;
use ir::Destroy;
use ir::Identifier;
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
pub struct Callee(MlirAttribute);

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
pub enum SymbolVisibility {
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

impl SymbolVisibility {
    // TODO: Is this attribute optional for `func.func` or is it empty on `None`?
    pub fn new_named_attribute(context: &Context, visibility: SymbolVisibility) -> Option<Named> {
        if visibility == SymbolVisibility::None {
            return None;
        }
        let s = StringBacked::from_string(&visibility.to_string());
        let string = StringAttr::new(context, &s.as_string_ref());
        let name = StringBacked::from_string(&"sym_visibility".to_string());
        let id = Identifier::new(context, &name.as_string_ref());
        let attr_named = Named::new(&id, &string.as_attribute());
        Some(attr_named)
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
        let attr_name = StringBacked::from_string(&"value".to_string());
        let attr_id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&attr_id, &callee.as_attribute());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(args);
        op_state.add_results(t);
        op_state.add_attributes(&[attr_named]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Call(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_callee(&self) -> Attribute {
        let attr_name = StringBacked::from_string(&"callee".to_string());
        self.as_operation().get_attribute_inherent(&attr_name.as_string_ref())
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
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
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
        let attr_name = StringBacked::from_string(&"value".to_string());
        let attr_id = Identifier::new(&context, &attr_name.as_string_ref());
        let attr_named = Named::new(&attr_id, &op.get_symbol_ref().as_attribute());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[attr_named]);
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

    pub fn get_value(&self) -> Attribute {
        let attr_name = StringBacked::from_string(&"value".to_string());
        self.as_operation().get_attribute_inherent(&attr_name.as_string_ref())
    }
}

impl Func {
    pub fn new(
        t: &Function,
        f_name: &StringRef,
        visibility: SymbolVisibility,
        attr_args: &[Attribute],
        attr_results: &[Attribute],
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
        let mut attrs: Vec<Named> = Vec::new();
        let sym_name_attr = StringAttr::new(&context, f_name);
        let sym_name_string = StringBacked::from_string(&"sym_name".to_string());
        let sym_name_id = Identifier::new(&context, &sym_name_string.as_string_ref());
        let sym_name_named = Named::new(&sym_name_id, &sym_name_attr.as_attribute());
        attrs.push(sym_name_named);
        let function_type_attr = TypeAttr::new(&t.as_type());
        let function_type_string = StringBacked::from_string(&"function_type".to_string());
        let function_type_id = Identifier::new(&context, &function_type_string.as_string_ref());
        let function_type_named = Named::new(&function_type_id, &function_type_attr.as_attribute());
        attrs.push(function_type_named);
        match SymbolVisibility::new_named_attribute(&context, visibility) {
            Some(named) => attrs.push(named),
            None        => {
                let mut region = Region::new();
                region.append_block(&mut block);
                op_state.add_regions(&[region]);
            },
        };
        let args_attr = Array::new(&context, attr_args);
        let args_string = StringBacked::from_string(&"arg_attrs".to_string());
        let args_id = Identifier::new(&context, &args_string.as_string_ref());
        let args_named = Named::new(&args_id, &args_attr.as_attribute());
        attrs.push(args_named);
        let results_attr = Array::new(&context, attr_results);
        let results_string = StringBacked::from_string(&"res_attrs".to_string());
        let results_id = Identifier::new(&context, &results_string.as_string_ref());
        let results_named = Named::new(&results_id, &results_attr.as_attribute());
        attrs.push(results_named);
        op_state.add_attributes(&attrs);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Func(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_arg_attributes(&self) -> Array {
        let attr_name = StringBacked::from_string(&"arg_attrs".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Array::from(*attr.get())
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_type(&self) -> Function {
        let attr_name = StringBacked::from_string(&"function_type".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        let attr_type = TypeAttr::from(*attr.get());
        Function::from(*attr_type.get_type().get())
    }

    pub fn get_symbol_ref(&self) -> SymbolRef {
        let attr_name = StringBacked::from_string(&"sym_name".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        let attr_string = StringAttr::from(*attr.get());
        SymbolRef::new_flat(&self.get_context(), &attr_string.get_string())
    }

    pub fn get_result_attributes(&self) -> Array {
        let attr_name = StringBacked::from_string(&"res_attrs".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        Array::from(*attr.get())
    }

    pub fn get_visibility(&self) -> SymbolVisibility {
        let attr_name = StringBacked::from_string(&"sym_visibility".to_string());
        let attr = self.as_operation().get_attribute_inherent(&attr_name.as_string_ref());
        let attr_string = StringAttr::from(*attr.get());
        match SymbolVisibility::from_str(attr_string.to_string().as_str()) {
            Ok(v)       => v,
            Err(msg)    => {
                eprintln!("{}", msg);
                exit(ExitCode::DialectError);
            },
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
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
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

impl Destroy for Call {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
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

impl cmp::PartialEq for Call {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl IRAttribute for Callee {
    fn as_attribute(&self) -> Attribute {
        Attribute::from(self.0)
    }

    fn get(&self) -> &MlirAttribute {
        &self.0
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl cmp::PartialEq for Callee {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

impl Destroy for CallIndirect {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
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

impl cmp::PartialEq for CallIndirect {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl Destroy for Constant {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
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

impl cmp::PartialEq for Constant {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl Destroy for Func {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
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

impl cmp::PartialEq for Func {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl IROp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl Destroy for Return {
    fn destroy(&mut self) -> () {
        self.as_operation().destroy()
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
            Trait::HasParent(&Op::Func),
            Trait::MemRefsNormalizable,
            Trait::ReturnLike,
            Trait::Terminator,
        ]
    }
}

impl cmp::PartialEq for Return {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl FromStr for SymbolVisibility {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            ""          => Ok(SymbolVisibility::None),
            "private"   => Ok(SymbolVisibility::Private),
            _           => Err(format!("Invalid symbol visibility: {}", s)),
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

impl fmt::Display for SymbolVisibility {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            SymbolVisibility::None      => "",
            SymbolVisibility::Private   => "private",
        })
    }
}
