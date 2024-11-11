// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use std::fmt;

use crate::dialects;
use dialects::IROp;

#[derive(Clone,Copy,PartialEq)]
pub enum Trait {
    AffineScope,
    AlwaysSpeculatableImplTrait,
    AutomaticAllocationScope,
    Commutative,
    ConstantLike,
    ElementWise,
    HasParent(&'static dyn IROp),
    IsolatedFromAbove,
    MemRefsNormalizable,
    ReturnLike,
    SameOperandAndResultType,
    Scalarizable,
    Tensorizable,
    Terminator,
    Vectorizable,
}

impl fmt::Display for Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Trait::AffineScope                  => "affine_scope".to_string(),
            Trait::AlwaysSpeculatableImplTrait  => "always_speculatable_impl_trait".to_string(),
            Trait::AutomaticAllocationScope     => "automatic_allocation_scope".to_string(),
            Trait::Commutative                  => "commutative".to_string(),
            Trait::ConstantLike                 => "constant_like".to_string(),
            Trait::ElementWise                  => "element_wise".to_string(),
            Trait::HasParent(op)                => format!("has_parent<{}>", op),
            Trait::IsolatedFromAbove            => "isolated_from_above".to_string(),
            Trait::MemRefsNormalizable          => "mem_refs_normalizable".to_string(),
            Trait::ReturnLike                   => "return_like".to_string(),
            Trait::SameOperandAndResultType     => "same_operand_and_result_type".to_string(),
            Trait::Scalarizable                 => "scalarizable".to_string(),
            Trait::Tensorizable                 => "tensorizable".to_string(),
            Trait::Terminator                   => "terminator".to_string(),
            Trait::Vectorizable                 => "vectorizable".to_string(),
        })
    }
}
