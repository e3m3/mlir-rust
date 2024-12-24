// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirOperation;

use std::ffi::c_uint;
use std::fmt;

use crate::attributes;
use crate::dialects;
use crate::effects;
use crate::exit_code;
use crate::interfaces;
use crate::ir;
use crate::traits;
use crate::types;

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::specialized::NamedI32DenseArray;
use attributes::specialized::NamedInitialization;
use attributes::specialized::NamedInteger;
use attributes::specialized::NamedMemorySpace;
use attributes::specialized::NamedPermutation;
use attributes::specialized::NamedString;
use attributes::specialized::NamedSymbolRef;
use attributes::specialized::NamedType;
use attributes::specialized::NamedUnit;
use attributes::symbol_ref::SymbolRef;
use dialects::IOp;
use dialects::IOperation;
use dialects::common::NonTemporal;
use dialects::common::OperandSegmentSizes;
use dialects::common::SymbolName;
use dialects::common::SymbolVisibility;
use dialects::common::SymbolVisibilityKind;
use effects::MEFF_NO_MEMORY_EFFECT;
use effects::MemoryEffectList;
use exit_code::ExitCode;
use exit_code::exit;
use interfaces::Interface;
use interfaces::MemoryEffectOpInterface;
use ir::Context;
use ir::Dialect;
use ir::Location;
use ir::OperationState;
use ir::Shape;
use ir::StringBacked;
use ir::Type;
use ir::Value;
use traits::Trait;
use types::IType;
use types::index::Index;
use types::integer::Integer as IntegerType;
use types::memref::MemRef;
use types::shaped::Shaped;
use types::unranked_memref::UnrankedMemRef;

///////////////////////////////
//  Attributes
///////////////////////////////

#[derive(Clone)]
pub struct Alignment(MlirAttribute);

#[derive(Clone)]
pub struct GlobalRef(MlirAttribute);

#[derive(Clone)]
pub struct GlobalType(MlirAttribute);

#[derive(Clone)]
pub struct InitialValue(MlirAttribute);

#[derive(Clone)]
pub struct IsConstant(MlirAttribute);

#[derive(Clone)]
pub struct Permutation(MlirAttribute);

///////////////////////////////
//  Enums
///////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Alloc,
    Alloca,
    AllocaScope,
    AllocaScopeReturn,
    AssumeAlignment,
    AtomicRMW,
    AtomicYield,
    Cast,
    CollapseShape,
    Copy,
    Dealloc,
    Dim,
    DmaStart,
    DmaWait,
    ExpandShape,
    ExtractAlignedPointerAsIndex,
    ExtractStridedMetadata,
    GenericAtomicRMW,
    GetGlobal,
    Global,
    MemorySpaceCast,
    Load,
    Prefetch,
    Rank,
    Realloc,
    ReinterpretCast,
    Reshape,
    Store,
    SubView,
    Transpose,
    View,
    Yield,
}

///////////////////////////////
//  Operations
///////////////////////////////

#[derive(Clone)]
pub struct Alloc(MlirOperation);

#[derive(Clone)]
pub struct Alloca(MlirOperation);

#[derive(Clone)]
pub struct Cast(MlirOperation);

#[derive(Clone)]
pub struct Copy(MlirOperation);

#[derive(Clone)]
pub struct Dealloc(MlirOperation);

#[derive(Clone)]
pub struct Dim(MlirOperation);

#[derive(Clone)]
pub struct GetGlobal(MlirOperation);

#[derive(Clone)]
pub struct Global(MlirOperation);

#[derive(Clone)]
pub struct Load(MlirOperation);

#[derive(Clone)]
pub struct Rank(MlirOperation);

#[derive(Clone)]
pub struct Store(MlirOperation);

#[derive(Clone)]
pub struct Transpose(MlirOperation);

#[derive(Clone)]
pub struct View(MlirOperation);

///////////////////////////////
//  Attribute Implementation
///////////////////////////////

impl Alignment {
    pub fn new(context: &Context, align: usize) -> Self {
        const WIDTH: usize = 64;
        <Self as NamedInteger>::new(context, align as i64, WIDTH)
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl GlobalRef {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl GlobalType {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl InitialValue {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl IsConstant {
    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }
}

impl Permutation {
    pub fn new(context: &Context, permutation: &[usize]) -> Self {
        let mut values: Vec<c_uint> = permutation.iter().map(|&v| v as c_uint).collect();
        <Self as NamedPermutation>::new(context, &mut values)
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
    fn get_name(&self) -> &'static str {
        match self {
            Op::Alloc => "alloc",
            Op::Alloca => "alloca",
            Op::AllocaScope => "alloca_scope",
            Op::AllocaScopeReturn => "alloca_scope.return",
            Op::AssumeAlignment => "assume_alignment",
            Op::AtomicRMW => "atomic_rmw",
            Op::AtomicYield => "atomic_yield",
            Op::Cast => "cast",
            Op::CollapseShape => "collapse_shape",
            Op::Copy => "copy",
            Op::Dealloc => "dealloc",
            Op::Dim => "dim",
            Op::DmaStart => "dma_start",
            Op::DmaWait => "dma_wait",
            Op::ExpandShape => "expand_shape",
            Op::ExtractAlignedPointerAsIndex => "extract_aligned_pointer_as_index",
            Op::ExtractStridedMetadata => "extract_strided_metadata",
            Op::GenericAtomicRMW => "generic_atomic_rmw",
            Op::GetGlobal => "get_global",
            Op::Global => "global",
            Op::MemorySpaceCast => "memory_space_cast",
            Op::Load => "load",
            Op::Prefetch => "prefetch",
            Op::Rank => "rank",
            Op::Realloc => "realloc",
            Op::ReinterpretCast => "reinterpret_cast",
            Op::Reshape => "reshape",
            Op::Store => "store",
            Op::SubView => "subview",
            Op::Transpose => "transpose",
            Op::View => "view",
            Op::Yield => "yield",
        }
    }
}

///////////////////////////////
//  Operation Implemention
///////////////////////////////

impl Alloc {
    pub fn new(
        t: &MemRef,
        dyn_sizes: &[Value],
        syms: &[Value],
        align: Option<&Alignment>,
        loc: &Location,
    ) -> Self {
        if dyn_sizes.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for dynamic sizes operands");
            exit(ExitCode::DialectError);
        }
        if syms.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for symbol operands");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let n_dyn_inputs = dyn_sizes.len() as i64;
        if let Some(n_dyn) = s.num_dynamic_dims() {
            if n_dyn != n_dyn_inputs {
                eprintln!(
                    "Expected number of dynamic sizes ({}) to match number \
                    of dynamic dimensions ({}) of the result memory reference type for alloc operation",
                    n_dyn_inputs, n_dyn,
                );
                exit(ExitCode::DialectError);
            }
        } else if !dyn_sizes.is_empty() {
            eprintln!(
                "Expected no dynamic sizes to match number of dynamic dimensions \
                of the result memory reference type for alloc operation"
            );
            exit(ExitCode::DialectError);
        }
        let n_syms = t.get_affine_map().num_symbols();
        let n_syms_inputs = syms.len() as isize;
        if n_syms != n_syms_inputs {
            eprintln!(
                "Expected number of symbols ({}) to match number of symbols in the affine map ({}) \
                of the result memory reference type for alloc operation",
                n_syms_inputs, n_syms,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Alloc.get_name(),
        ));
        let mut args: Vec<Value> = Vec::new();
        args.append(&mut dyn_sizes.to_vec());
        args.append(&mut syms.to_vec());
        let opseg_attr =
            OperandSegmentSizes::new(&context, &[dyn_sizes.len() as i32, syms.len() as i32]);
        let mut attrs = vec![opseg_attr.as_named_attribute()];
        if let Some(align_attr) = align {
            attrs.push(align_attr.as_named_attribute());
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&attrs);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Alloc(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_alignment(&self) -> Option<Alignment> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(Alignment::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(Alignment::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Alloca {
    pub fn new(
        t: &MemRef,
        dyn_sizes: &[Value],
        syms: &[Value],
        align: Option<&Alignment>,
        loc: &Location,
    ) -> Self {
        if dyn_sizes.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for dynamic sizes operands");
            exit(ExitCode::DialectError);
        }
        if syms.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for symbol operands");
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        if let Some(n_dyn) = s.num_dynamic_dims() {
            if n_dyn != dyn_sizes.len() as i64 {
                eprintln!(
                    "Expected number of dynamic sizes to match number of dynamic dimensions \
                    of the result memory reference type for alloca operation"
                );
                exit(ExitCode::DialectError);
            }
        } else if !dyn_sizes.is_empty() {
            eprintln!(
                "Expected number of dynamic sizes to match number of dynamic dimensions \
                of the result memory reference type for alloca operation"
            );
            exit(ExitCode::DialectError);
        }
        if t.get_affine_map().num_symbols() != syms.len() as isize {
            eprintln!(
                "Expected number of symbols to match number of symbols in the affine map \
                of the result memory reference type for alloca operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Alloca.get_name(),
        ));
        let mut args: Vec<Value> = Vec::new();
        args.append(&mut dyn_sizes.to_vec());
        args.append(&mut syms.to_vec());
        let opseg_attr =
            OperandSegmentSizes::new(&context, &[dyn_sizes.len() as i32, syms.len() as i32]);
        let mut attrs = vec![opseg_attr.as_named_attribute()];
        if let Some(align_attr) = align {
            attrs.push(align_attr.as_named_attribute());
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&attrs);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Alloca(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_alignment(&self) -> Option<Alignment> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(Alignment::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(Alignment::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Cast {
    fn new(t: &Type, source: &Value, loc: &Location) -> Self {
        let context = t.get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Cast.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone()]);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    /// TODO: The following properties (1 and 2) are not checked by this constructor [1]:
    /// a.  Both are ranked memref types with the same element type, address space, and rank and:
    ///     1.  Both have the same layout or both have compatible strided layouts.
    ///     2.  The individual sizes (resp. offset and strides in the case of strided memrefs)
    ///         may convert constant dimensions to dynamic dimensions and vice-versa.
    /// [1]: https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcast-memrefcastop
    pub fn new_ranked<T: NamedMemorySpace>(t: &MemRef, source: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        let is_ranked = t_source.is_memref();
        let is_unranked = t_source.is_unranked_memref();
        if !is_ranked && !is_unranked {
            eprintln!(
                "Expected ranked or unranked memory reference for source operand \
                of cast operation"
            );
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_source = Shaped::from(*t_source.get());
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!(
                "Expected matching element types for source and target operands of cast operation"
            );
            exit(ExitCode::DialectError);
        }
        if is_ranked {
            let t_ranked = MemRef::from(*t_source.get());
            if t.get_memory_space::<T>() != t_ranked.get_memory_space::<T>() {
                eprintln!(
                    "Expected matching memory space for source and target operands \
                    of cast operation"
                );
                exit(ExitCode::DialectError);
            }
            let rank = s.rank().unwrap_or(-1);
            let rank_source = s_source.rank().unwrap_or(-1);
            if rank != rank_source {
                eprintln!(
                    "Expected matching ranks for ranked memory reference \
                    source ({}) and target ({}) operands of cast operation",
                    rank_source, rank,
                );
                exit(ExitCode::DialectError);
            }
        } else {
            let t_unranked = UnrankedMemRef::from(*t_source.get());
            if t.get_memory_space::<T>() != t_unranked.get_memory_space::<T>() {
                eprintln!(
                    "Expected matching memory space for source and target operands \
                    of cast operation"
                );
                exit(ExitCode::DialectError);
            }
        }
        Self::new(&t.as_type(), source, loc)
    }

    pub fn new_unranked<T: NamedMemorySpace>(
        t: &UnrankedMemRef,
        source: &Value,
        loc: &Location,
    ) -> Self {
        let t_source = source.get_type();
        let is_ranked = t_source.is_memref();
        let is_unranked = t_source.is_unranked_memref();
        if !is_ranked && !is_unranked {
            eprintln!(
                "Expected ranked or unranked memory reference for source operand \
                of cast operation"
            );
            exit(ExitCode::DialectError);
        }
        let s = t.as_shaped();
        let s_source = Shaped::from(*t_source.get());
        if s.get_element_type() != s_source.get_element_type() {
            eprintln!(
                "Expected matching element types for source and target operands of copy operation"
            );
            exit(ExitCode::DialectError);
        }
        if is_ranked {
            let t_ranked = MemRef::from(*t_source.get());
            if t.get_memory_space::<T>() != t_ranked.get_memory_space::<T>() {
                eprintln!(
                    "Expected matching memory space for source and target operands \
                    of cast operation"
                );
                exit(ExitCode::DialectError);
            }
        } else {
            let t_unranked = UnrankedMemRef::from(*t_source.get());
            if t.get_memory_space::<T>() != t_unranked.get_memory_space::<T>() {
                eprintln!(
                    "Expected matching memory space for source and target operands \
                    of ast operation"
                );
                exit(ExitCode::DialectError);
            }
        }
        Self::new(&t.as_type(), source, loc)
    }

    pub fn from(op: MlirOperation) -> Self {
        Cast(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Copy {
    pub fn new(context: &Context, source: &Value, target: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        let t_target = target.get_type();
        if !t_source.is_memref() && !t_source.is_unranked_memref() {
            eprintln!(
                "Expected ranked or unranked memory reference for source operand \
                of copy operation"
            );
            exit(ExitCode::DialectError);
        }
        if !t_target.is_memref() && !t_target.is_unranked_memref() {
            eprintln!(
                "Expected ranked or unranked memory reference for target operand \
                of copy operation"
            );
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*t_source.get());
        let s_target = Shaped::from(*t_target.get());
        if s_source.get_element_type() != s_target.get_element_type() {
            eprintln!(
                "Expected matching element types for source and target operands of copy operation"
            );
            exit(ExitCode::DialectError);
        }
        if s_source.unpack() != s_target.unpack() {
            eprintln!("Expected matching shapes for source and target operands of copy operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Copy.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone(), target.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Copy(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Dealloc {
    pub fn new(context: &Context, value: &Value, loc: &Location) -> Self {
        if !value.get_type().is_memref() {
            eprintln!("Expected memory reference type for operand of deallocation operation");
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Dealloc.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[value.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Dealloc(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Dim {
    /// The behavior for when the index is out of bounds is undefined; do not handle as the value
    /// is not statically known.
    pub fn new(context: &Context, source: &Value, index: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        let is_unranked = t_source.is_unranked_memref();
        let is_non_0_ranked =
            t_source.is_memref() && Shaped::from(*t_source.get()).rank().unwrap_or(0) > 0;
        if !is_unranked && !is_non_0_ranked {
            eprintln!(
                "Expected unranked or non-0-ranked memory reference type \
                for source operand of dim operaton"
            );
            exit(ExitCode::DialectError);
        }
        if !index.get_type().is_index() {
            eprintln!("Expected index type for index operand of dim operaton");
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context);
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Dim.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone(), index.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Dim(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl GetGlobal {
    pub fn new(t: &MemRef, global_ref: &GlobalRef, loc: &Location) -> Self {
        if !t.as_shaped().is_static() {
            eprintln!(
                "Expected statically shaped memory reference type for result \
                of get global operation"
            );
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::GetGlobal.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[global_ref.as_named_attribute()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        GetGlobal(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Global {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: &GlobalType,
        sym_name: &SymbolName,
        visibility: SymbolVisibilityKind,
        init: Option<&InitialValue>,
        is_const: Option<&IsConstant>,
        align: Option<&Alignment>,
        loc: &Location,
    ) -> Self {
        let context = t.get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Global.get_name(),
        ));
        let mut attrs = vec![sym_name.as_named_attribute(), t.as_named_attribute()];
        if let Some(visibility_) = SymbolVisibility::new(&context, visibility) {
            attrs.push(visibility_.as_named_attribute());
        }
        if let Some(init_) = init {
            attrs.push(init_.as_named_attribute());
        }
        if let Some(is_const_) = is_const {
            attrs.push(is_const_.as_named_attribute());
        }
        if let Some(align_) = align {
            attrs.push(align_.as_named_attribute());
        }
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&attrs);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Global(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_alignment(&self) -> Option<Alignment> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(Alignment::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(Alignment::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_constant_attribute(&self) -> Option<IsConstant> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(IsConstant::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(IsConstant::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_global_ref(&self, context: &Context) -> GlobalRef {
        GlobalRef::new(context, &self.get_symbol_ref().get_value().unwrap())
    }

    pub fn get_initial_value(&self) -> Option<InitialValue> {
        let op = self.as_operation();
        let attr_name = StringBacked::from(InitialValue::get_name());
        let s_ref = attr_name.as_string_ref();
        if op.has_attribute_inherent(&s_ref) {
            let attr = op.get_attribute_inherent(&s_ref);
            Some(InitialValue::from(*attr.get()))
        } else {
            None
        }
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }

    pub fn get_symbol_name(&self) -> SymbolName {
        let attr_name = StringBacked::from(SymbolName::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        SymbolName::from(*attr.get())
    }

    pub fn get_symbol_ref(&self) -> SymbolRef {
        SymbolRef::new_flat(
            &self.get_context(),
            &self.get_symbol_name().as_string().get_string(),
        )
    }

    pub fn get_symbol_visibility(&self) -> SymbolVisibility {
        let attr_name = StringBacked::from(SymbolVisibility::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        SymbolVisibility::from(*attr.get())
    }

    pub fn get_type_attribute(&self) -> GlobalType {
        let attr_name = StringBacked::from(GlobalType::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        GlobalType::from(*attr.get())
    }

    pub fn is_constant(&self) -> bool {
        self.get_constant_attribute().is_none()
    }

    pub fn is_declaration(&self) -> bool {
        self.get_initial_value().is_none()
    }

    pub fn is_initialized(&self) -> bool {
        match self.get_initial_value() {
            None => false,
            Some(v) => v.is_initialized(),
        }
    }
}

impl Load {
    pub fn new(
        t: &Type,
        source: &Value,
        indices: &[Value],
        is_nt: &NonTemporal,
        loc: &Location,
    ) -> Self {
        if !source.get_type().is_memref() {
            eprintln!("Expected ranked memory reference type for source operand of load operation");
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for index operand(s) of load operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*source.get_type().get());
        if *t != s_source.get_element_type() {
            eprintln!(
                "Expected matching types for source element type and result of load operation"
            );
            exit(ExitCode::DialectError);
        }
        let n_indices = indices.len() as i64;
        let rank_source = s_source.rank().unwrap_or(0);
        if rank_source != n_indices {
            eprintln!(
                "Expected matching arity of indices ({}) and source memory reference rank ({}) \
                of load operation",
                n_indices, rank_source,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Load.get_name(),
        ));
        let mut args = vec![source.clone()];
        args.append(&mut indices.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[is_nt.as_named_attribute()]);
        op_state.add_operands(&args);
        op_state.add_results(&[t.clone()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Load(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_non_temporal_attribute(&self) -> NonTemporal {
        let attr_name = StringBacked::from(NonTemporal::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        NonTemporal::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Rank {
    pub fn new(context: &Context, source: &Value, loc: &Location) -> Self {
        let t_source = source.get_type();
        let is_unranked = t_source.is_unranked_memref();
        let is_ranked = t_source.is_memref();
        if !is_unranked && !is_ranked {
            eprintln!(
                "Expected unranked or ranked memory reference type \
                for source operand of rank operaton"
            );
            exit(ExitCode::DialectError);
        }
        let t = Index::new(context);
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Rank.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&[source.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Rank(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl Store {
    pub fn new(
        context: &Context,
        value: &Value,
        target: &Value,
        indices: &[Value],
        is_nt: &NonTemporal,
        loc: &Location,
    ) -> Self {
        if !target.get_type().is_memref() {
            eprintln!(
                "Expected ranked memory reference type for target operand of store operation"
            );
            exit(ExitCode::DialectError);
        }
        if indices.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for index operand(s) of store operation");
            exit(ExitCode::DialectError);
        }
        let s_target = Shaped::from(*target.get_type().get());
        if value.get_type() != s_target.get_element_type() {
            eprintln!(
                "Expected matching types for target element type and result of store operation"
            );
            exit(ExitCode::DialectError);
        }
        let rank_target = s_target.rank().unwrap_or(0);
        let n_indices = indices.len() as i64;
        if rank_target != n_indices {
            eprintln!(
                "Expected matching arity of indices ({}) and target memory reference rank ({}) \
                of store operation",
                n_indices, rank_target,
            );
            exit(ExitCode::DialectError);
        }
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Store.get_name(),
        ));
        let mut args = vec![value.clone(), target.clone()];
        args.append(&mut indices.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[is_nt.as_named_attribute()]);
        op_state.add_operands(&args);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Store(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }
}

impl Transpose {
    pub fn new(t: &MemRef, source: &Value, p: &Permutation, loc: &Location) -> Self {
        if !source.get_type().is_memref() {
            eprintln!(
                "Expected ranked memory reference type for source operand of transpose operation"
            );
            exit(ExitCode::DialectError);
        }
        let t_source = MemRef::from(*source.get_type().get());
        if t.as_shaped().get_element_type() != t_source.as_shaped().get_element_type() {
            eprintln!(
                "Expected matching element types for source operand and \
                result memory reference types of transpose operation"
            );
            exit(ExitCode::DialectError);
        }
        let n_result = t.get_affine_map().num_dims();
        let n_perm = p.as_affine_map().num_dims();
        let n_source = t_source.get_affine_map().num_dims();
        if n_result != n_perm || n_result != n_source {
            eprintln!(
                "Expected matching number of dimensions for result ({}), source ({}), \
                and permutation ({}) of transpose operation",
                n_result, n_source, n_perm,
            );
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::Transpose.get_name(),
        ));
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_attributes(&[p.as_named_attribute()]);
        op_state.add_operands(&[source.clone()]);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        Transpose(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_permutation(&self) -> Permutation {
        let attr_name = StringBacked::from(Permutation::get_name());
        let attr = self
            .as_operation()
            .get_attribute_inherent(&attr_name.as_string_ref());
        Permutation::from(*attr.get())
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

impl View {
    pub fn new(
        t: &MemRef,
        source: &Value,
        byte_shift: &Value,
        sizes: &[Value],
        loc: &Location,
    ) -> Self {
        if !source.get_type().is_memref() {
            eprintln!("Expected ranked memory reference type for source operand of view operation");
            exit(ExitCode::DialectError);
        }
        let s_source = Shaped::from(*source.get_type().get());
        let rank_source = s_source.rank().unwrap_or(-1);
        if rank_source != 1 {
            eprintln!(
                "Expected 1-D ranked memory reference type for source operand ({}) \
                of view operation",
                rank_source,
            );
            exit(ExitCode::DialectError);
        }
        let t_elem = s_source.get_element_type();
        if !t_elem.is_integer() {
            eprintln!(
                "Expected integer element memory reference type for source operand \
                of view operation"
            );
            exit(ExitCode::DialectError);
        }
        const WIDTH: usize = 8;
        let t_int = IntegerType::from(*t_elem.get());
        let width_elem = t_int.get_width();
        if !t_int.is_signless() || width_elem != WIDTH {
            eprintln!(
                "Expected signless {}-bit integer elements for memory reference type \
                for source operand ({}-bit elements) of view operation",
                WIDTH, width_elem,
            );
            exit(ExitCode::DialectError);
        }
        if !byte_shift.get_type().is_index() {
            eprintln!("Expected index type for byte shift operand of view operation");
            exit(ExitCode::DialectError);
        }
        if sizes.iter().any(|v| !v.get_type().is_index()) {
            eprintln!("Expected index type for size operand(s) of view operation");
            exit(ExitCode::DialectError);
        }
        let context = t.as_type().get_context();
        let dialect = context.get_dialect_memref();
        let name = StringBacked::from(format!(
            "{}.{}",
            dialect.get_namespace(),
            Op::View.get_name(),
        ));
        let mut args = vec![source.clone(), byte_shift.clone()];
        args.append(&mut sizes.to_vec());
        let mut op_state = OperationState::new(&name.as_string_ref(), loc);
        op_state.add_operands(&args);
        op_state.add_results(&[t.as_type()]);
        Self::from(*op_state.create_operation().get())
    }

    pub fn from(op: MlirOperation) -> Self {
        View(op)
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_result(&self) -> Value {
        self.as_operation().get_result(0)
    }
}

///////////////////////////////
//  Trait Implemention
///////////////////////////////

impl From<MlirAttribute> for Alignment {
    fn from(attr: MlirAttribute) -> Self {
        Alignment(attr)
    }
}

impl IAttribute for Alignment {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for Alignment {
    fn get_name() -> &'static str {
        "alignment"
    }
}

impl NamedInteger for Alignment {}

impl IOperation for Alloc {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::OpAsmOpInterface]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Alloc.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Alloc
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AttrSizedOperandSegments]
    }
}

impl IOperation for Alloca {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::DestructurableAllocationOpInterface,
            Interface::OpAsmOpInterface,
            Interface::PromotableAllocationOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Alloca.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Alloca
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AttrSizedOperandSegments]
    }
}

impl IOperation for Cast {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::CastOpInterface,
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::ViewLikeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Cast.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Cast
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[
            Trait::AlwaysSpeculatableImplTrait,
            Trait::SameOperandsAndResultType,
            Trait::MemRefsNormalizable,
        ]
    }
}

impl IOperation for Copy {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::CopyOpInterface]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Copy.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Copy
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::SameOperandsElementType, Trait::SameOperandsShape]
    }
}

impl IOperation for Dealloc {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Dealloc.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Dealloc
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl IOperation for Dim {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::ShapedDimOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Dim.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Dim
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl IOperation for GetGlobal {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::SymbolUserOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::GetGlobal.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::GetGlobal
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl IOperation for Global {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[Interface::Symbol]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Global.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Global
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[]
    }
}

impl From<MlirAttribute> for InitialValue {
    fn from(attr: MlirAttribute) -> Self {
        InitialValue(attr)
    }
}

impl IAttribute for InitialValue {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for InitialValue {
    fn get_name() -> &'static str {
        "initial_value"
    }
}

impl NamedInitialization for InitialValue {}

impl From<MlirAttribute> for IsConstant {
    fn from(attr: MlirAttribute) -> Self {
        IsConstant(attr)
    }
}

impl IAttribute for IsConstant {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for IsConstant {
    fn get_name() -> &'static str {
        "constant"
    }
}

impl NamedUnit for IsConstant {}

impl IOperation for Load {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::DestructurableAllocationOpInterface,
            Interface::InferTypeOpInterface,
            Interface::PromotableAllocationOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Load.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Load
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl IOp for Op {
    fn get_name(&self) -> &'static str {
        self.get_name()
    }
}

impl From<MlirAttribute> for Permutation {
    fn from(attr: MlirAttribute) -> Self {
        Permutation(attr)
    }
}

impl IAttribute for Permutation {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for Permutation {
    fn get_name() -> &'static str {
        "permutation"
    }
}

impl NamedPermutation for Permutation {}

impl IOperation for Rank {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::InferTypeOpInterface,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Rank.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Rank
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl IOperation for Store {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::DestructurableAllocationOpInterface,
            Interface::PromotableAllocationOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Store.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Store
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::MemRefsNormalizable]
    }
}

impl IOperation for Transpose {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::Transpose.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::Transpose
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

impl From<MlirAttribute> for GlobalRef {
    fn from(attr: MlirAttribute) -> Self {
        GlobalRef(attr)
    }
}

impl IAttribute for GlobalRef {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for GlobalRef {
    fn get_name() -> &'static str {
        "name"
    }
}

impl NamedSymbolRef for GlobalRef {}

impl From<MlirAttribute> for GlobalType {
    fn from(attr: MlirAttribute) -> Self {
        GlobalType(attr)
    }
}

impl IAttribute for GlobalType {
    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl IAttributeNamed for GlobalType {
    fn get_name() -> &'static str {
        "type"
    }
}

impl NamedType for GlobalType {}

impl IOperation for View {
    fn get(&self) -> &MlirOperation {
        self.get()
    }

    fn get_dialect(&self) -> Dialect {
        self.as_operation().get_context().get_dialect_memref()
    }

    fn get_effects(&self) -> MemoryEffectList {
        &[MEFF_NO_MEMORY_EFFECT]
    }

    fn get_interfaces(&self) -> &'static [Interface] {
        &[
            Interface::ConditionallySpeculatable,
            Interface::MemoryEffect(MemoryEffectOpInterface::NoMemoryEffect),
            Interface::OpAsmOpInterface,
            Interface::ViewLikeOpInterface,
        ]
    }

    fn get_mut(&mut self) -> &mut MlirOperation {
        self.get_mut()
    }

    fn get_name(&self) -> &'static str {
        Op::View.get_name()
    }

    fn get_op(&self) -> &'static dyn IOp {
        &Op::View
    }

    fn get_traits(&self) -> &'static [Trait] {
        &[Trait::AlwaysSpeculatableImplTrait]
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Op::Alloc => "AllocOp",
            Op::Alloca => "AllocaOp",
            Op::AllocaScope => "AllocaScopeOp",
            Op::AllocaScopeReturn => "AllocaScopeReturnOp",
            Op::AssumeAlignment => "AssumeAlignmentOp",
            Op::AtomicRMW => "AtomicRMWOp",
            Op::AtomicYield => "AtomicYieldOp",
            Op::Cast => "CastOp",
            Op::CollapseShape => "CollapseShapeOp",
            Op::Copy => "CopyOp",
            Op::Dealloc => "DeallocOp",
            Op::Dim => "DimOp",
            Op::DmaStart => "DmaStartOp",
            Op::DmaWait => "DmaWaitOp",
            Op::ExpandShape => "ExpandShapeOp",
            Op::ExtractAlignedPointerAsIndex => "ExtractAlignedPointerAsIndexOp",
            Op::ExtractStridedMetadata => "ExtractStridedMetadataOp",
            Op::GenericAtomicRMW => "GenericAtomicRMWOp",
            Op::GetGlobal => "GetGlobalOp",
            Op::Global => "GlobalOp",
            Op::MemorySpaceCast => "MemorySpaceCastOp",
            Op::Load => "LoadOp",
            Op::Prefetch => "PrefetchOp",
            Op::Rank => "RankOp",
            Op::Realloc => "ReallocOp",
            Op::ReinterpretCast => "ReinterpretCastOp",
            Op::Reshape => "ReshapeOp",
            Op::Store => "StoreOp",
            Op::SubView => "SubViewOp",
            Op::Transpose => "TransposeOp",
            Op::View => "ViewOp",
            Op::Yield => "YieldOp",
        })
    }
}
