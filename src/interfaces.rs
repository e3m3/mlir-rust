// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use std::fmt;

#[derive(Clone,Copy,PartialEq)]
pub enum Interface {
    ArithFastMathInterface,
    ArithIntegerOverflowFlagsInterface,
    CallOpInterface,
    CallableOpInterface,
    CastOpInterface,
    ConditionallySpeculatable,
    CopyOpInterface,
    DestinationStyleOpInterface,
    DestructurableAllocationOpInterface,
    FunctionOpInterface,
    InferIntRangeInterface,
    InferTypeOpInterface,
    LinalgContractionOpInterface,
    LinalgStructuredInterface,
    MaskableOpInterface,
    MemoryEffect(MemoryEffectOpInterface),
    OffsetSizeAndStrideOpInterface,
    OpAsmOpInterface,
    PromotableAllocationOpInterface,
    RegionBranchTerminatorOpInterface,
    ReifyRankedShapeTypeOpInterface,
    ShapedDimOpInterface,
    Symbol,
    SymbolUserOpInterface,
    VectorTransferOpInterface,
    VectorUnrollOpInterface,
    ViewLikeOpInterface,
}

#[derive(Clone,Copy,PartialEq)]
pub enum MemoryEffectOpInterface {
    NoMemoryEffect,
    MemoryEffect,
    RecursiveMemoryEffects,
    UndefinedMemoryEffect,
}

impl fmt::Display for Interface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Interface::ArithFastMathInterface               => "arith_fast_math_interface".to_string(),
            Interface::ArithIntegerOverflowFlagsInterface   => {
                "arith_integer_overflow_flags_interface".to_string()
            },
            Interface::CallOpInterface                      => "call_op_interface".to_string(),
            Interface::CallableOpInterface                  => "callable_op_interface".to_string(),
            Interface::CastOpInterface                      => "cast_op_interface".to_string(),
            Interface::ConditionallySpeculatable            => "conditionally_speculatable".to_string(),
            Interface::CopyOpInterface                      => "copy_op_interface".to_string(),
            Interface::DestinationStyleOpInterface          => "destination_style_op_interface".to_string(),
            Interface::DestructurableAllocationOpInterface  => {
                "destructurable_allocation_op_interface".to_string()
            },
            Interface::FunctionOpInterface                  => "function_op_interface".to_string(),
            Interface::InferIntRangeInterface               => "infer_int_range_interface".to_string(),
            Interface::InferTypeOpInterface                 => "infer_type_op_interface".to_string(),
            Interface::LinalgContractionOpInterface         => {
                "linalg_contraction_op_interface".to_string()
            },
            Interface::LinalgStructuredInterface            => "linalg_structured_interface".to_string(),
            Interface::MaskableOpInterface                  => "maskable_op_interface".to_string(),
            Interface::MemoryEffect(e)                      => format!("memory_effect_op_interface({})", e),
            Interface::OffsetSizeAndStrideOpInterface       => {
                "offset_size_and_stride_op_interface".to_string()
            },
            Interface::OpAsmOpInterface                     => "op_asm_op_interface".to_string(),
            Interface::PromotableAllocationOpInterface      => {
                "promotable_allocation_op_interface".to_string()
            },
            Interface::RegionBranchTerminatorOpInterface    => {
                "region_branch_terminator_op_interface".to_string()
            },
            Interface::ReifyRankedShapeTypeOpInterface      => {
                "reify_ranked_shape_type_op_interface".to_string()
            },
            Interface::ShapedDimOpInterface                 => "shaped_dim_op_interface".to_string(),
            Interface::Symbol                               => "symbol".to_string(),
            Interface::SymbolUserOpInterface                => "symbol_user_op_interface".to_string(),
            Interface::VectorTransferOpInterface            => "vector_transfer_op_interface".to_string(),
            Interface::VectorUnrollOpInterface              => "vector_unroll_op_interface".to_string(),
            Interface::ViewLikeOpInterface                  => "view_like_op_interface".to_string(),
        })
    }
}

impl fmt::Display for MemoryEffectOpInterface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            MemoryEffectOpInterface::NoMemoryEffect         => "no_memory_effect",
            MemoryEffectOpInterface::MemoryEffect           => "memory_effect",
            MemoryEffectOpInterface::RecursiveMemoryEffects => "recursive_memory_effects",
            MemoryEffectOpInterface::UndefinedMemoryEffect  => "undefined_memory_effect",
        })
    }
}
