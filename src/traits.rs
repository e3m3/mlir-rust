// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::collections::HashSet;
use std::fmt;

use crate::dialects;
use crate::do_unsafe;

use dialects::OpRef;
use dialects::StaticOpList;
use dialects::static_op_list_to_string;

/// Expresses a 1:1 binary relationship between an Op and a Trait.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Bool {
    False(OpRef, Trait),
    True(OpRef, Trait),
}

/// Expresses a Many:Many binary relationship between Ops and Traits.
#[derive(Clone, Eq, PartialEq)]
pub enum BoolSet {
    False(HashSet<OpRef>, HashSet<Trait>),
    True(HashSet<OpRef>, HashSet<Trait>),
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Trait {
    AffineScope,
    AlwaysSpeculatableImplTrait,
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    Commutative,
    ConstantLike,
    ElementWise,
    HasParent(StaticOpList),
    Idempotent,
    InferTypeOpAdaptor,
    IsolatedFromAbove,
    MemRefsNormalizable,
    NoRegionArguments,
    RecursiveMemoryEffects,
    RecursivelySpeculatableImplTrait,
    ReturnLike,
    SameOperandsAndResultType,
    SameOperandsElementType,
    SameOperandsShape,
    SameTypeOperands,
    Scalarizable,
    SingleBlock,
    SingleBlockImplicitTerminator(StaticOpList),
    Tensorizable,
    Terminator,
    Vectorizable,
}

impl Bool {
    pub fn new_false(op: OpRef, t: Trait) -> Self {
        Self::False(op, t)
    }

    pub fn new_true(op: OpRef, t: Trait) -> Self {
        Self::True(op, t)
    }

    pub fn get_op(&self) -> &OpRef {
        match self {
            Self::False(op, _) => op,
            Self::True(op, _) => op,
        }
    }

    pub fn get_trait(&self) -> &Trait {
        match self {
            Self::False(_, t) => t,
            Self::True(_, t) => t,
        }
    }
}

impl BoolSet {
    pub fn new_false(ops_: &[OpRef], ts_: &[Trait]) -> Self {
        let mut ops = HashSet::<OpRef>::default();
        let mut ts = HashSet::<Trait>::default();
        ops_.iter().for_each(|&op| {
            let _ = ops.insert(op);
        });
        ts_.iter().for_each(|&t| {
            let _ = ts.insert(t);
        });
        Self::False(ops, ts)
    }

    pub fn new_true(ops_: &[OpRef], ts_: &[Trait]) -> Self {
        let mut ops = HashSet::<OpRef>::default();
        let mut ts = HashSet::<Trait>::default();
        ops_.iter().for_each(|&op| {
            let _ = ops.insert(op);
        });
        ts_.iter().for_each(|&t| {
            let _ = ts.insert(t);
        });
        Self::True(ops, ts)
    }

    pub fn get_ops(&self) -> &HashSet<OpRef> {
        match self {
            Self::False(ops, _) => ops,
            Self::True(ops, _) => ops,
        }
    }

    pub fn get_ops_mut(&mut self) -> &mut HashSet<OpRef> {
        let ops = (self.get_ops() as *const HashSet<OpRef>).cast_mut();
        match do_unsafe!(ops.as_mut()) {
            Some(ops) => ops,
            None => {
                panic!("Failed to get mutable reference to operations in set");
            }
        }
    }

    pub fn get_traits(&self) -> &HashSet<Trait> {
        match self {
            Self::False(_, ts) => ts,
            Self::True(_, ts) => ts,
        }
    }

    pub fn get_traits_mut(&mut self) -> &mut HashSet<Trait> {
        let ts = (self.get_traits() as *const HashSet<Trait>).cast_mut();
        match do_unsafe!(ts.as_mut()) {
            Some(ts) => ts,
            None => {
                panic!("Failed to get mutable reference to traits in set");
            }
        }
    }
}

impl fmt::Display for Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Trait::AffineScope => "affine_scope".to_string(),
            Trait::AlwaysSpeculatableImplTrait => "always_speculatable_impl_trait".to_string(),
            Trait::AttrSizedOperandSegments => "attr_sized_operand_segments".to_string(),
            Trait::AutomaticAllocationScope => "automatic_allocation_scope".to_string(),
            Trait::Commutative => "commutative".to_string(),
            Trait::ConstantLike => "constant_like".to_string(),
            Trait::ElementWise => "element_wise".to_string(),
            Trait::HasParent(ops) => {
                let s = static_op_list_to_string(ops);
                format!("has_parent<{}>", s)
            }
            Trait::Idempotent => "Idempotent".to_string(),
            Trait::InferTypeOpAdaptor => "infer_type_op_adaptor".to_string(),
            Trait::IsolatedFromAbove => "isolated_from_above".to_string(),
            Trait::MemRefsNormalizable => "memrefs_normalizable".to_string(),
            Trait::NoRegionArguments => "no_region_arguments".to_string(),
            Trait::RecursiveMemoryEffects => "recursive_memory_effects".to_string(),
            Trait::RecursivelySpeculatableImplTrait =>
                "recursively_speculatable_impl_trait".to_string(),
            Trait::ReturnLike => "return_like".to_string(),
            Trait::SameOperandsAndResultType => "same_operands_and_result_type".to_string(),
            Trait::SameOperandsElementType => "same_operands_element_type".to_string(),
            Trait::SameOperandsShape => "same_operands_shape".to_string(),
            Trait::SameTypeOperands => "same_type_operands".to_string(),
            Trait::Scalarizable => "scalarizable".to_string(),
            Trait::SingleBlock => "single_block".to_string(),
            Trait::SingleBlockImplicitTerminator(ops) => {
                let s = static_op_list_to_string(ops);
                format!("has_parent<{}>", s)
            }
            Trait::Tensorizable => "tensorizable".to_string(),
            Trait::Terminator => "terminator".to_string(),
            Trait::Vectorizable => "vectorizable".to_string(),
        })
    }
}
