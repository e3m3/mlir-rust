// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::fmt;

use crate::dialects;
use dialects::IOp;

pub type StaticOpList = &'static [&'static dyn IOp];

#[derive(Clone, Copy, PartialEq)]
pub enum Trait {
    AffineScope,
    AlwaysSpeculatableImplTrait,
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    Commutative,
    ConstantLike,
    ElementWise,
    HasParent(StaticOpList),
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
    Scalarizable,
    SingleBlock,
    SingleBlockImplicitTerminator(StaticOpList),
    Tensorizable,
    Terminator,
    Vectorizable,
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
            Trait::InferTypeOpAdaptor => "infer_type_op_adaptor".to_string(),
            Trait::IsolatedFromAbove => "isolated_from_above".to_string(),
            Trait::MemRefsNormalizable => "memrefs_normalizable".to_string(),
            Trait::NoRegionArguments => "no_region_arguments".to_string(),
            Trait::RecursiveMemoryEffects => "recursive_memory_effects".to_string(),
            Trait::RecursivelySpeculatableImplTrait => "recursively_speculatable_impl_trait".to_string(),
            Trait::ReturnLike => "return_like".to_string(),
            Trait::SameOperandsAndResultType => "same_operands_and_result_type".to_string(),
            Trait::SameOperandsElementType => "same_operands_element_type".to_string(),
            Trait::SameOperandsShape => "same_operands_shape".to_string(),
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

fn static_op_list_to_string(ops: StaticOpList) -> String {
    let s_vec: Vec<String> = ops.iter().map(|op| op.to_string()).collect();
    s_vec.join(",")
}
