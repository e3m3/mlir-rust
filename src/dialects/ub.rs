// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirOperation;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::specialized::CustomAttributeData;
use attributes::specialized::NamedOpaque;
use dialects::IROp;
use dialects::IROperation;
use effects::MemoryEffectList;
use effects::MEFF_NO_MEMORY_EFFECT;
use exit_code::exit;
use exit_code::ExitCode;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::OperationState;
use ir::StringBacked;
use traits::Trait;
use types::integer::Integer as IntegerType;
use types::IRType;
use types::none::None as NoneType;

use std::fmt;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct PoisonValue(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone,Copy,PartialEq)]
pub enum Op {
    Poison  = 0,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Poison(MlirOperation);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl PoisonValue {
    pub fn new(context: &Context) -> Self {
        let t = NoneType::new(context).as_type();
        let cad = CustomAttributeData::new("poison".to_string(), "ub".to_string(), vec![]);
        <Self as NamedOpaque>::new_custom(&t, &cad)
    }

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
            Op::Poison  => "poison",
        }
    }
}

///////////////////////////////
//  Operation Implemention
///////////////////////////////

impl Poison {
    pub fn new(context: &Context, loc: &Location) -> Self {
        let t = IntegerType::new(context, 32);
        let dialect = Self::get_dialect(context);
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Poison.get_name(),
        ));
        let value = PoisonValue::new(context);
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[value.as_named_attribute()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    fn get_dialect(context: &Context) -> Dialect {
        match context.load_dialect("ub") {
            Some(d) => d,
            None    => {
                eprintln!("Failed to load ub dialect");
                exit(ExitCode::DialectError);
            },
        }
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

///////////////////////////////
//  Trait Implemention
///////////////////////////////

impl IROp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirOperation> for Poison {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl IROperation for Poison {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        Self::get_dialect(&self.as_operation().get_context())
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
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Poison.get_name()
    }

    fn get_op(&self) -> &'static dyn IROp {
        &Op::Poison
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::ConstantLike,
        ]
    }
}

impl From<MlirAttribute> for PoisonValue {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IRAttribute for PoisonValue {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IRAttributeNamed for PoisonValue {
    fn get_name() -> &'static str {
        "value"
    }
}

impl NamedOpaque for PoisonValue {}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Poison  => "PoisonOp",
        })
    }
}
