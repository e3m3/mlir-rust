// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;
use mlir_sys::MlirBlock;
use mlir_sys::MlirContext;
use mlir_sys::MlirDialect;
use mlir_sys::MlirDialectRegistry;
use mlir_sys::MlirIdentifier;
use mlir_sys::MlirLocation;
use mlir_sys::MlirLogicalResult;
use mlir_sys::MlirModule;
use mlir_sys::MlirNamedAttribute;
use mlir_sys::MlirOpOperand;
use mlir_sys::MlirOperation;
use mlir_sys::MlirOperationState;
use mlir_sys::MlirPass;
use mlir_sys::MlirRegion;
use mlir_sys::MlirStringCallback;
use mlir_sys::MlirStringRef;
use mlir_sys::MlirSymbolTable;
use mlir_sys::MlirType;
use mlir_sys::MlirTypeID;
use mlir_sys::MlirValue;
use mlir_sys::mlirAttributeDump;
use mlir_sys::mlirAttributeEqual;
use mlir_sys::mlirAttributeGetContext;
use mlir_sys::mlirAttributeGetDialect;
use mlir_sys::mlirAttributeGetNull;
use mlir_sys::mlirAttributeGetType;
use mlir_sys::mlirAttributeGetTypeID;
use mlir_sys::mlirAttributeIsAAffineMap;
use mlir_sys::mlirAttributeIsAArray;
use mlir_sys::mlirAttributeIsABool;
use mlir_sys::mlirAttributeIsADenseBoolArray;
use mlir_sys::mlirAttributeIsADenseElements;
use mlir_sys::mlirAttributeIsADenseF32Array;
use mlir_sys::mlirAttributeIsADenseF64Array;
use mlir_sys::mlirAttributeIsADenseFPElements;
use mlir_sys::mlirAttributeIsADenseI8Array;
use mlir_sys::mlirAttributeIsADenseI16Array;
use mlir_sys::mlirAttributeIsADenseI32Array;
use mlir_sys::mlirAttributeIsADenseI64Array;
use mlir_sys::mlirAttributeIsADenseIntElements;
use mlir_sys::mlirAttributeIsADenseResourceElements;
use mlir_sys::mlirAttributeIsADictionary;
use mlir_sys::mlirAttributeIsAElements;
use mlir_sys::mlirAttributeIsAFlatSymbolRef;
use mlir_sys::mlirAttributeIsAFloat;
use mlir_sys::mlirAttributeIsAInteger;
use mlir_sys::mlirAttributeIsAIntegerSet;
use mlir_sys::mlirAttributeIsALocation;
use mlir_sys::mlirAttributeIsAOpaque;
use mlir_sys::mlirAttributeIsASparseElements;
use mlir_sys::mlirAttributeIsAStridedLayout;
use mlir_sys::mlirAttributeIsAString;
use mlir_sys::mlirAttributeIsASymbolRef;
use mlir_sys::mlirAttributeIsAType;
use mlir_sys::mlirAttributeIsAUnit;
use mlir_sys::mlirAttributeParseGet;
use mlir_sys::mlirAttributePrint;
use mlir_sys::mlirBlockAddArgument;
use mlir_sys::mlirBlockAppendOwnedOperation;
use mlir_sys::mlirBlockArgumentGetArgNumber;
use mlir_sys::mlirBlockArgumentGetOwner;
use mlir_sys::mlirBlockArgumentSetType;
use mlir_sys::mlirBlockCreate;
use mlir_sys::mlirBlockDestroy;
use mlir_sys::mlirBlockDetach;
use mlir_sys::mlirBlockEqual;
use mlir_sys::mlirBlockEraseArgument;
use mlir_sys::mlirBlockGetArgument;
use mlir_sys::mlirBlockGetFirstOperation;
use mlir_sys::mlirBlockGetNextInRegion;
use mlir_sys::mlirBlockGetNumArguments;
use mlir_sys::mlirBlockGetParentOperation;
use mlir_sys::mlirBlockGetParentRegion;
use mlir_sys::mlirBlockGetTerminator;
use mlir_sys::mlirBlockInsertArgument;
use mlir_sys::mlirBlockInsertOwnedOperation;
use mlir_sys::mlirBlockInsertOwnedOperationAfter;
use mlir_sys::mlirBlockInsertOwnedOperationBefore;
use mlir_sys::mlirBlockPrint;
use mlir_sys::mlirContextAppendDialectRegistry;
use mlir_sys::mlirContextCreate;
use mlir_sys::mlirContextCreateWithRegistry;
use mlir_sys::mlirContextCreateWithThreading;
use mlir_sys::mlirContextDestroy;
use mlir_sys::mlirContextEnableMultithreading;
use mlir_sys::mlirContextGetAllowUnregisteredDialects;
use mlir_sys::mlirContextGetNumLoadedDialects;
use mlir_sys::mlirContextGetNumRegisteredDialects;
use mlir_sys::mlirContextGetOrLoadDialect;
use mlir_sys::mlirContextIsRegisteredOperation;
use mlir_sys::mlirContextLoadAllAvailableDialects;
use mlir_sys::mlirContextSetAllowUnregisteredDialects;
use mlir_sys::mlirDialectEqual;
use mlir_sys::mlirDialectGetContext;
use mlir_sys::mlirDialectGetNamespace;
use mlir_sys::mlirDialectHandleInsertDialect;
use mlir_sys::mlirDialectHandleLoadDialect;
use mlir_sys::mlirDialectRegistryCreate;
use mlir_sys::mlirDialectRegistryDestroy;
use mlir_sys::mlirDisctinctAttrCreate;
use mlir_sys::mlirGetDialectHandle__arith__;
use mlir_sys::mlirGetDialectHandle__func__;
use mlir_sys::mlirGetDialectHandle__gpu__;
use mlir_sys::mlirGetDialectHandle__linalg__;
use mlir_sys::mlirGetDialectHandle__llvm__;
use mlir_sys::mlirGetDialectHandle__memref__;
use mlir_sys::mlirGetDialectHandle__shape__;
use mlir_sys::mlirGetDialectHandle__spirv__;
use mlir_sys::mlirGetDialectHandle__tensor__;
use mlir_sys::mlirGetDialectHandle__vector__;
use mlir_sys::mlirIdentifierEqual;
use mlir_sys::mlirIdentifierGet;
use mlir_sys::mlirIdentifierGetContext;
use mlir_sys::mlirIdentifierStr;
use mlir_sys::mlirLocationCallSiteGet;
use mlir_sys::mlirLocationEqual;
use mlir_sys::mlirLocationFileLineColGet;
use mlir_sys::mlirLocationFromAttribute;
use mlir_sys::mlirLocationGetAttribute;
use mlir_sys::mlirLocationGetContext;
use mlir_sys::mlirLocationPrint;
use mlir_sys::mlirLocationUnknownGet;
use mlir_sys::mlirModuleCreateEmpty;
use mlir_sys::mlirModuleCreateParse;
use mlir_sys::mlirModuleDestroy;
use mlir_sys::mlirModuleFromOperation;
use mlir_sys::mlirModuleGetBody;
use mlir_sys::mlirModuleGetOperation;
use mlir_sys::mlirOpOperandGetNextUse;
use mlir_sys::mlirOpOperandGetOperandNumber;
use mlir_sys::mlirOpOperandGetValue;
use mlir_sys::mlirOpOperandIsNull;
use mlir_sys::mlirOpResultGetOwner;
use mlir_sys::mlirOperationClone;
use mlir_sys::mlirOperationCreate;
use mlir_sys::mlirOperationCreateParse;
use mlir_sys::mlirOperationDestroy;
use mlir_sys::mlirOperationDump;
use mlir_sys::mlirOperationEqual;
use mlir_sys::mlirOperationGetBlock;
use mlir_sys::mlirOperationGetContext;
use mlir_sys::mlirOperationGetDiscardableAttribute;
use mlir_sys::mlirOperationGetDiscardableAttributeByName;
use mlir_sys::mlirOperationGetFirstRegion;
use mlir_sys::mlirOperationGetInherentAttributeByName;
use mlir_sys::mlirOperationGetLocation;
use mlir_sys::mlirOperationGetName;
use mlir_sys::mlirOperationGetNextInBlock;
use mlir_sys::mlirOperationGetNumDiscardableAttributes;
use mlir_sys::mlirOperationGetNumOperands;
use mlir_sys::mlirOperationGetNumRegions;
use mlir_sys::mlirOperationGetNumResults;
use mlir_sys::mlirOperationGetNumSuccessors;
use mlir_sys::mlirOperationGetOperand;
use mlir_sys::mlirOperationGetParentOperation;
use mlir_sys::mlirOperationGetRegion;
use mlir_sys::mlirOperationGetResult;
use mlir_sys::mlirOperationGetSuccessor;
use mlir_sys::mlirOperationGetTypeID;
use mlir_sys::mlirOperationHasInherentAttributeByName;
use mlir_sys::mlirOperationMoveAfter;
use mlir_sys::mlirOperationMoveBefore;
use mlir_sys::mlirOperationPrint;
use mlir_sys::mlirOperationRemoveDiscardableAttributeByName;
use mlir_sys::mlirOperationRemoveFromParent;
use mlir_sys::mlirOperationSetDiscardableAttributeByName;
use mlir_sys::mlirOperationSetInherentAttributeByName;
use mlir_sys::mlirOperationSetOperand;
use mlir_sys::mlirOperationSetOperands;
use mlir_sys::mlirOperationSetSuccessor;
use mlir_sys::mlirOperationStateAddAttributes;
use mlir_sys::mlirOperationStateAddOperands;
use mlir_sys::mlirOperationStateAddOwnedRegions;
use mlir_sys::mlirOperationStateAddResults;
use mlir_sys::mlirOperationStateAddSuccessors;
use mlir_sys::mlirOperationStateEnableResultTypeInference;
use mlir_sys::mlirOperationStateGet;
use mlir_sys::mlirOperationVerify;
use mlir_sys::mlirRegionAppendOwnedBlock;
use mlir_sys::mlirRegionCreate;
use mlir_sys::mlirRegionDestroy;
use mlir_sys::mlirRegionEqual;
use mlir_sys::mlirRegionGetFirstBlock;
use mlir_sys::mlirRegionGetNextInOperation;
use mlir_sys::mlirRegionInsertOwnedBlock;
use mlir_sys::mlirRegionInsertOwnedBlockAfter;
use mlir_sys::mlirRegionInsertOwnedBlockBefore;
use mlir_sys::mlirRegionTakeBody;
use mlir_sys::mlirRegisterAllDialects;
use mlir_sys::mlirRegisterAllPasses;
use mlir_sys::mlirStringRefCreateFromCString;
use mlir_sys::mlirStringRefEqual;
use mlir_sys::mlirSymbolTableCreate;
use mlir_sys::mlirSymbolTableDestroy;
use mlir_sys::mlirSymbolTableErase;
use mlir_sys::mlirSymbolTableGetSymbolAttributeName;
use mlir_sys::mlirSymbolTableGetVisibilityAttributeName;
use mlir_sys::mlirSymbolTableInsert;
use mlir_sys::mlirSymbolTableLookup;
use mlir_sys::mlirSymbolTableReplaceAllSymbolUses;
use mlir_sys::mlirTypeDump;
use mlir_sys::mlirTypeEqual;
use mlir_sys::mlirTypeGetContext;
use mlir_sys::mlirTypeGetDialect;
use mlir_sys::mlirTypeGetTypeID;
use mlir_sys::mlirTypeIDCreate;
use mlir_sys::mlirTypeIDEqual;
use mlir_sys::mlirTypeIDHashValue;
use mlir_sys::mlirTypeIsAComplex;
use mlir_sys::mlirTypeIsAFloat;
use mlir_sys::mlirTypeIsAFunction;
use mlir_sys::mlirTypeIsAIndex;
use mlir_sys::mlirTypeIsAInteger;
use mlir_sys::mlirTypeIsAMemRef;
use mlir_sys::mlirTypeIsANone;
use mlir_sys::mlirTypeIsAOpaque;
use mlir_sys::mlirTypeIsARankedTensor;
use mlir_sys::mlirTypeIsAShaped;
use mlir_sys::mlirTypeIsATensor;
use mlir_sys::mlirTypeIsATuple;
use mlir_sys::mlirTypeIsAUnrankedMemRef;
use mlir_sys::mlirTypeIsAUnrankedTensor;
use mlir_sys::mlirTypeIsAVector;
use mlir_sys::mlirTypeParseGet;
use mlir_sys::mlirTypePrint;
use mlir_sys::mlirValueDump;
use mlir_sys::mlirValueEqual;
use mlir_sys::mlirValueGetFirstUse;
use mlir_sys::mlirValueGetType;
use mlir_sys::mlirValueIsABlockArgument;
use mlir_sys::mlirValueIsAOpResult;
use mlir_sys::mlirValuePrint;
use mlir_sys::mlirValueReplaceAllUsesOfWith;
use mlir_sys::mlirValueSetType;

use std::cmp;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::fmt;
use std::mem;
use std::ptr;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::types;

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::named::Named;
use dialects::IOp;
use exit_code::ExitCode;
use exit_code::exit;
use types::GetWidth;
use types::IType;
use types::IsPromotableTo;
use types::unit::Unit;

///////////////////////////////
//  Constants and Macros
///////////////////////////////

const STATE_BUFFER_LENGTH: usize = 4096;
const STATE_BUFFER_DATA_LENGTH: usize = STATE_BUFFER_LENGTH - mem::size_of::<usize>();

macro_rules! print_method {
    ($FunctionName:ident) => {
        pub fn print(&self, state: &mut StringCallbackState) -> () {
            let callback = StringCallback::new();
            unsafe {
                $FunctionName(*self.get(), *callback.get(), state.as_void_mut_ptr());
            }
            let idx = state.num_bytes_written();
            state.get_data_mut()[idx] = b'\0';
        }
    };
}
pub(crate) use print_method;

///////////////////////////////
//  Traits
///////////////////////////////

pub trait Destroy {
    fn destroy(&mut self) -> ();
}

pub trait Shape {
    fn rank(&self) -> isize;
    fn get(&self, i: isize) -> i64;

    fn to_vec(&self) -> Vec<i64> {
        if self.rank() >= 0 {
            (0..self.rank()).map(|i| self.get(i)).collect()
        } else {
            Vec::new()
        }
    }

    fn to_vec_transpose(&self) -> Vec<i64> {
        self.to_vec().into_iter().rev().collect()
    }

    fn unpack(&self) -> ShapeUnpacked {
        (self.rank(), self.to_vec())
    }

    fn unpack_transpose(&self) -> ShapeUnpacked {
        (self.rank(), self.to_vec_transpose())
    }
}
pub type ShapeUnpacked = (isize, Vec<i64>);

///////////////////////////////
//  IR
///////////////////////////////

#[derive(Clone)]
pub struct Attribute(MlirAttribute);

#[derive(Clone)]
pub struct Block(MlirBlock, usize);

pub struct BlockIter<'a>(&'a Block, Option<Operation>);

#[derive(Clone)]
pub struct Context(MlirContext);

#[derive(Clone)]
pub struct Dialect(MlirDialect);

#[derive(Clone)]
pub struct Identifier(MlirIdentifier);

#[derive(Clone)]
pub struct Location(MlirLocation);

#[derive(Clone)]
pub struct LogicalResult(MlirLogicalResult);

#[derive(Clone)]
pub struct Module(MlirModule);

#[derive(Clone)]
pub struct Pass(MlirPass);

pub struct Operation(MlirOperation);

pub struct OperationIter<'a>(&'a Operation, Option<Region>);

#[derive(Clone)]
pub struct OperationState(MlirOperationState);

#[derive(Clone)]
pub struct OpOperand(MlirOpOperand);

#[derive(Clone)]
pub struct Region(MlirRegion, usize);

pub struct RegionIter<'a>(&'a Region, Option<Block>);

#[derive(Clone)]
pub struct Registry(MlirDialectRegistry);

#[derive(Clone)]
pub struct ShapeImpl<T: Clone + Sized>(T);

#[derive(Clone)]
pub struct StringBacked(MlirStringRef, CString);

#[derive(Clone)]
pub struct StringCallback(MlirStringCallback);
pub type StringCallbackFn = unsafe extern "C" fn(MlirStringRef, *mut c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StringCallbackState {
    bytes_written: usize,
    data: [u8; STATE_BUFFER_DATA_LENGTH],
}

#[derive(Clone)]
pub struct StringRef(MlirStringRef);

#[derive(Clone)]
pub struct SymbolTable(MlirSymbolTable);

#[derive(Clone)]
pub struct Type(MlirType);

#[derive(Clone)]
pub struct TypeID(MlirTypeID);

#[derive(Clone)]
pub struct Value(MlirValue);

pub struct ValueIter<'a>(&'a Value, Option<OpOperand>);

///////////////////////////////
//  IR Implementation
///////////////////////////////

impl Attribute {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirAttributeGetNull()))
    }

    /// This is the only mention of DisctinctAttr.
    /// Also, why is it mispelled in the C API?
    pub fn new_distinct(attr: &Attribute) -> Self {
        Self::from(do_unsafe!(mlirDisctinctAttrCreate(*attr.get())))
    }

    pub fn from_parse(context: &Context, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirAttributeParseGet(*context.get(), *s.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirAttributeDump(self.0))
    }

    pub fn get(&self) -> &MlirAttribute {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirAttributeGetContext(self.0)))
    }

    pub fn get_dialect(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirAttributeGetDialect(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirAttribute {
        &mut self.0
    }

    pub fn get_type(&self) -> Type {
        Type::from(do_unsafe!(mlirAttributeGetType(self.0)))
    }

    pub fn get_type_id(&self) -> TypeID {
        TypeID::from(do_unsafe!(mlirAttributeGetTypeID(self.0)))
    }

    pub fn is_affine_map(&self) -> bool {
        do_unsafe!(mlirAttributeIsAAffineMap(self.0))
    }

    pub fn is_array(&self) -> bool {
        do_unsafe!(mlirAttributeIsAArray(self.0))
    }

    pub fn is_bool(&self) -> bool {
        do_unsafe!(mlirAttributeIsABool(self.0))
    }

    pub fn is_dense_array_bool(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseBoolArray(self.0))
    }

    pub fn is_dense_array_f32(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseF32Array(self.0))
    }

    pub fn is_dense_array_f64(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseF64Array(self.0))
    }

    pub fn is_dense_array_i8(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseI8Array(self.0))
    }

    pub fn is_dense_array_i16(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseI16Array(self.0))
    }

    pub fn is_dense_array_i32(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseI32Array(self.0))
    }

    pub fn is_dense_array_i64(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseI64Array(self.0))
    }

    pub fn is_dense_elements(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseElements(self.0))
    }

    pub fn is_dense_elements_float(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseFPElements(self.0))
    }

    pub fn is_dense_elements_int(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseIntElements(self.0))
    }

    pub fn is_dense_elements_resource(&self) -> bool {
        do_unsafe!(mlirAttributeIsADenseResourceElements(self.0))
    }

    pub fn is_dictionary(&self) -> bool {
        do_unsafe!(mlirAttributeIsADictionary(self.0))
    }

    pub fn is_elements(&self) -> bool {
        do_unsafe!(mlirAttributeIsAElements(self.0))
    }

    pub fn is_flat_symbol_ref(&self) -> bool {
        do_unsafe!(mlirAttributeIsAFlatSymbolRef(self.0))
    }

    pub fn is_float(&self) -> bool {
        do_unsafe!(mlirAttributeIsAFloat(self.0))
    }

    pub fn is_index(&self) -> bool {
        self.get_type().is_index()
    }

    pub fn is_integer(&self) -> bool {
        do_unsafe!(mlirAttributeIsAInteger(self.0))
    }

    pub fn is_integer_set(&self) -> bool {
        do_unsafe!(mlirAttributeIsAIntegerSet(self.0))
    }

    pub fn is_location(&self) -> bool {
        do_unsafe!(mlirAttributeIsALocation(self.0))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn is_opaque(&self) -> bool {
        do_unsafe!(mlirAttributeIsAOpaque(self.0))
    }

    pub fn is_sparse_elements(&self) -> bool {
        do_unsafe!(mlirAttributeIsASparseElements(self.0))
    }

    pub fn is_strided_layout(&self) -> bool {
        do_unsafe!(mlirAttributeIsAStridedLayout(self.0))
    }

    pub fn is_string(&self) -> bool {
        do_unsafe!(mlirAttributeIsAString(self.0))
    }

    pub fn is_symbol_ref(&self) -> bool {
        do_unsafe!(mlirAttributeIsASymbolRef(self.0))
    }

    pub fn is_type(&self) -> bool {
        do_unsafe!(mlirAttributeIsAType(self.0))
    }

    pub fn is_unit(&self) -> bool {
        do_unsafe!(mlirAttributeIsAUnit(self.0))
    }

    print_method!(mlirAttributePrint);

    pub fn to_location(&self) -> Location {
        Location::from(do_unsafe!(mlirLocationFromAttribute(self.0)))
    }
}

impl Block {
    pub fn new(num_args: isize, args: &[Type], locs: &[Location]) -> Self {
        assert!(num_args == args.len() as isize);
        assert!(num_args == locs.len() as isize);
        let args_raw: Vec<MlirType> = args.iter().map(|a| *a.get()).collect();
        let locs_raw: Vec<MlirLocation> = locs.iter().map(|m| *m.get()).collect();
        Self::from(do_unsafe!(mlirBlockCreate(
            num_args,
            args_raw.as_ptr(),
            locs_raw.as_ptr()
        )))
    }

    pub fn new_empty() -> Self {
        Self::new(0, &[], &[])
    }

    pub fn add_arg(&mut self, t: &Type, loc: &Location) -> Value {
        Value::from(do_unsafe!(mlirBlockAddArgument(
            *self.get_mut(),
            *t.get(),
            *loc.get()
        )))
    }

    pub fn append_operation(&mut self, op: &mut Operation) -> () {
        do_unsafe!(mlirBlockAppendOwnedOperation(
            *self.get_mut(),
            *op.get_mut()
        ));
        *self.num_operations_mut() += 1;
    }

    pub fn detach(&mut self) -> () {
        do_unsafe!(mlirBlockDetach(self.0))
    }

    pub fn erase_arg(&mut self, i: usize) -> () {
        do_unsafe!(mlirBlockEraseArgument(self.0, i as c_uint))
    }

    pub fn get(&self) -> &MlirBlock {
        &self.0
    }

    pub fn get_arg(&self, i: isize) -> Value {
        if i >= self.num_args() || i < 0 {
            eprintln!("Argument index '{}' out of bounds for block", i);
            exit(ExitCode::IRError);
        }
        Value::from(do_unsafe!(mlirBlockGetArgument(self.0, i)))
    }

    pub fn get_mut(&mut self) -> &mut MlirBlock {
        &mut self.0
    }

    pub fn get_parent(&self) -> Region {
        Region::from(do_unsafe!(mlirBlockGetParentRegion(self.0)))
    }

    pub fn get_parent_operation(&self) -> Operation {
        Operation::from(do_unsafe!(mlirBlockGetParentOperation(self.0)))
    }

    pub fn get_terminator(&self) -> Operation {
        Operation::from(do_unsafe!(mlirBlockGetTerminator(self.0)))
    }

    pub fn insert_arg(&mut self, t: &Type, loc: &Location, i: usize) -> Value {
        Value::from(do_unsafe!(mlirBlockInsertArgument(
            *self.get_mut(),
            i as isize,
            *t.get(),
            *loc.get(),
        )))
    }

    pub fn insert_operation(&mut self, op: &mut Operation, i: usize) -> () {
        do_unsafe!(mlirBlockInsertOwnedOperation(
            *self.get_mut(),
            i as isize,
            *op.get_mut()
        ));
        *self.num_operations_mut() += 1;
    }

    pub fn insert_operation_after(&mut self, anchor: &Operation, op: &mut Operation) -> () {
        do_unsafe!(mlirBlockInsertOwnedOperationAfter(
            *self.get_mut(),
            *anchor.get(),
            *op.get_mut()
        ));
        *self.num_operations_mut() += 1;
    }

    pub fn insert_operation_before(&mut self, anchor: &Operation, op: &mut Operation) -> () {
        do_unsafe!(mlirBlockInsertOwnedOperationBefore(
            *self.get_mut(),
            *anchor.get(),
            *op.get_mut()
        ));
        *self.num_operations_mut() += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.num_operations() == 0 && self.iter().next().is_none()
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn iter(&self) -> BlockIter {
        BlockIter(self, None)
    }

    print_method!(mlirBlockPrint);

    pub fn num_args(&self) -> isize {
        do_unsafe!(mlirBlockGetNumArguments(self.0))
    }

    fn __num_operations(&self) -> usize {
        self.iter().fold(0, |acc, _op| acc + 1) as usize
    }

    pub fn num_operations(&self) -> usize {
        self.1
    }

    pub fn num_operations_mut(&mut self) -> &mut usize {
        &mut self.1
    }
}

impl Context {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirContextCreate()))
    }

    pub fn new_with_threading_enabled(allow_threading: bool) -> Self {
        Self::from(do_unsafe!(mlirContextCreateWithThreading(allow_threading)))
    }

    pub fn from_registry(registry: &Registry) -> Self {
        Self::from(do_unsafe!(mlirContextCreateWithRegistry(
            *registry.get(),
            false
        )))
    }

    pub fn append_registry(&mut self, registry: &Registry) -> () {
        do_unsafe!(mlirContextAppendDialectRegistry(
            *self.get_mut(),
            *registry.get()
        ))
    }

    pub fn get(&self) -> &MlirContext {
        &self.0
    }

    pub fn get_dialect_arith(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__arith__(),
            self.0
        )))
    }

    pub fn get_dialect_func(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__func__(),
            self.0
        )))
    }

    pub fn get_dialect_gpu(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__gpu__(),
            self.0
        )))
    }

    pub fn get_dialect_linalg(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__linalg__(),
            self.0
        )))
    }

    pub fn get_dialect_llvm(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__llvm__(),
            self.0
        )))
    }

    pub fn get_dialect_memref(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__memref__(),
            self.0
        )))
    }

    pub fn get_dialect_shape(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__shape__(),
            self.0
        )))
    }

    pub fn get_dialect_spirv(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__spirv__(),
            self.0
        )))
    }

    pub fn get_dialect_tensor(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__tensor__(),
            self.0
        )))
    }

    pub fn get_dialect_vector(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirDialectHandleLoadDialect(
            mlirGetDialectHandle__vector__(),
            self.0
        )))
    }

    pub fn get_mut(&mut self) -> &mut MlirContext {
        &mut self.0
    }

    pub fn get_unknown_location(&self) -> Location {
        Location::new_unknown(self)
    }

    pub fn is_allowed_unregistered_dialects(&self) -> bool {
        do_unsafe!(mlirContextGetAllowUnregisteredDialects(*self.get()))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn is_registered_operation(&self, s: &StringRef) -> bool {
        do_unsafe!(mlirContextIsRegisteredOperation(*self.get(), *s.get()))
    }

    pub fn load_all_available_dialects(&mut self) -> () {
        do_unsafe!(mlirContextLoadAllAvailableDialects(*self.get_mut()))
    }

    /// Load a registered dialect with name
    pub fn load_dialect(&self, name: &str) -> Option<Dialect> {
        let string = StringBacked::from_str(name).unwrap();
        let dialect = do_unsafe!(mlirContextGetOrLoadDialect(*self.get(), *string.get()));
        if dialect.ptr.is_null() {
            None
        } else {
            Some(Dialect::from(dialect))
        }
    }

    pub fn num_loaded_dialects(&self) -> isize {
        do_unsafe!(mlirContextGetNumLoadedDialects(*self.get()))
    }

    pub fn num_registered_dialects(&self) -> isize {
        do_unsafe!(mlirContextGetNumRegisteredDialects(*self.get()))
    }

    pub fn set_allow_unregistered_dialects(&mut self, allow: bool) -> () {
        do_unsafe!(mlirContextSetAllowUnregisteredDialects(
            *self.get_mut(),
            allow
        ))
    }

    pub fn set_enable_multithreading(&mut self, allow: bool) -> () {
        do_unsafe!(mlirContextEnableMultithreading(*self.get_mut(), allow))
    }
}

impl Dialect {
    pub fn get(&self) -> &MlirDialect {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirDialectGetContext(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirDialect {
        &mut self.0
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirDialectGetNamespace(self.0)))
    }

    pub fn get_op_name(&self, op: &dyn IOp) -> StringBacked {
        StringBacked::from(format!("{}.{}", self.get_namespace(), op.get_name(),))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }
}

impl Identifier {
    pub fn new(context: &Context, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirIdentifierGet(*context.get(), *s.get())))
    }

    pub fn as_string(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirIdentifierStr(self.0)))
    }

    pub fn get(&self) -> &MlirIdentifier {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirIdentifierGetContext(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirIdentifier {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }
}

impl Location {
    pub fn new(context: &Context, name: &StringRef, line: usize, col: usize) -> Self {
        Self::from(do_unsafe!(mlirLocationFileLineColGet(
            *context.get(),
            *name.get(),
            line as c_uint,
            col as c_uint,
        )))
    }

    pub fn new_unknown(context: &Context) -> Self {
        Location::from(do_unsafe!(mlirLocationUnknownGet(*context.get())))
    }

    pub fn from_call_site(callee: &Location, caller: &Location) -> Self {
        Self::from(do_unsafe!(mlirLocationCallSiteGet(
            *callee.get(),
            *caller.get()
        )))
    }

    pub fn get(&self) -> &MlirLocation {
        &self.0
    }

    pub fn get_attribute(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirLocationGetAttribute(self.0)))
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirLocationGetContext(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirLocation {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    print_method!(mlirLocationPrint);
}

impl LogicalResult {
    pub fn get(&self) -> &MlirLogicalResult {
        &self.0
    }

    pub fn get_bool(&self) -> bool {
        self.get_value() != 0
    }

    pub fn get_mut(&mut self) -> &mut MlirLogicalResult {
        &mut self.0
    }

    pub fn get_value(&self) -> i8 {
        self.0.value
    }
}

impl Module {
    pub fn as_operation(&self) -> Operation {
        Operation::from(do_unsafe!(mlirModuleGetOperation(self.0)))
    }

    pub fn new(loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirModuleCreateEmpty(*loc.get())))
    }

    pub fn from_parse(context: &Context, string: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirModuleCreateParse(
            *context.get(),
            *string.get()
        )))
    }

    pub fn from_operation(op: &Operation) -> Self {
        Self::from(do_unsafe!(mlirModuleFromOperation(*op.get())))
    }

    pub fn get(&self) -> &MlirModule {
        &self.0
    }

    pub fn get_body(&self) -> Block {
        Block::from(do_unsafe!(mlirModuleGetBody(self.0)))
    }

    pub fn get_context(&self) -> Context {
        self.as_operation().get_context()
    }

    pub fn get_mut(&mut self) -> &mut MlirModule {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn take_body(&mut self, block: &mut Block) -> () {
        let mut region = self.as_operation().get_region(0);
        let mut region_temp = Region::new();
        if !block.get_parent().is_null() {
            block.detach();
        }
        region_temp.append_block(block);
        region.take_body(&mut region_temp);
    }
}

impl Operation {
    pub fn new(state: &mut OperationState) -> Self {
        Self::from(do_unsafe!(mlirOperationCreate(state.get_mut())))
    }

    pub fn from_parse(context: &Context, op: &StringRef, src_name: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirOperationCreateParse(
            *context.get(),
            *op.get(),
            *src_name.get()
        )))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirOperationDump(self.0))
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_attribute_discardable(&self, name: &StringRef) -> Attribute {
        Attribute::from(do_unsafe!(mlirOperationGetDiscardableAttributeByName(
            self.0,
            *name.get()
        )))
    }

    pub fn get_attribute_discardable_at(&self, i: usize) -> Named {
        Named::from(do_unsafe!(mlirOperationGetDiscardableAttribute(
            self.0, i as isize
        )))
    }

    pub fn get_attribute_inherent(&self, name: &StringRef) -> Attribute {
        Attribute::from(do_unsafe!(mlirOperationGetInherentAttributeByName(
            self.0,
            *name.get()
        )))
    }

    pub fn get_block(&self) -> Block {
        Block::from(do_unsafe!(mlirOperationGetBlock(self.0)))
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirOperationGetContext(self.0)))
    }

    pub fn get_location(&self) -> Location {
        Location::from(do_unsafe!(mlirOperationGetLocation(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirOperation {
        &mut self.0
    }

    pub fn get_name(&self) -> Identifier {
        Identifier::from(do_unsafe!(mlirOperationGetName(self.0)))
    }

    pub fn get_operand(&self, i: isize) -> Value {
        Value::from(do_unsafe!(mlirOperationGetOperand(self.0, i)))
    }

    pub fn get_parent(&self) -> Operation {
        Operation::from(do_unsafe!(mlirOperationGetParentOperation(self.0)))
    }

    pub fn get_region(&self, i: isize) -> Region {
        if i >= self.num_regions() || i < 0 {
            eprint!("Region index '{}' out of bounds for operation: ", i);
            self.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        Region::from(do_unsafe!(mlirOperationGetRegion(self.0, i)))
    }

    pub fn get_result(&self, i: isize) -> Value {
        Value::from(do_unsafe!(mlirOperationGetResult(self.0, i)))
    }

    pub fn get_successor(&self, i: isize) -> Block {
        Block::from(do_unsafe!(mlirOperationGetSuccessor(self.0, i)))
    }

    pub fn get_symbol_table(&self) -> Option<SymbolTable> {
        let table = do_unsafe!(mlirSymbolTableCreate(self.0));
        if table.ptr.is_null() {
            None
        } else {
            Some(SymbolTable::from(table))
        }
    }

    pub fn get_type_id(&self) -> TypeID {
        TypeID::from(do_unsafe!(mlirOperationGetTypeID(self.0)))
    }

    pub fn has_attribute_inherent(&self, name: &StringRef) -> bool {
        do_unsafe!(mlirOperationHasInherentAttributeByName(self.0, *name.get()))
    }

    pub fn insert_after(&mut self, other: &Self) -> () {
        do_unsafe!(mlirOperationMoveAfter(*self.get_mut(), other.0))
    }

    pub fn insert_before(&mut self, other: &Self) -> () {
        do_unsafe!(mlirOperationMoveBefore(*self.get_mut(), other.0))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn iter(&self) -> OperationIter {
        OperationIter(self, None)
    }

    pub fn num_attributes_discardable(&self) -> isize {
        do_unsafe!(mlirOperationGetNumDiscardableAttributes(self.0))
    }

    pub fn num_operands(&self) -> isize {
        do_unsafe!(mlirOperationGetNumOperands(self.0))
    }

    pub fn num_regions(&self) -> isize {
        do_unsafe!(mlirOperationGetNumRegions(self.0))
    }

    pub fn num_results(&self) -> isize {
        do_unsafe!(mlirOperationGetNumResults(self.0))
    }

    pub fn num_successors(&self) -> isize {
        do_unsafe!(mlirOperationGetNumSuccessors(self.0))
    }

    print_method!(mlirOperationPrint);

    pub fn remove_attribute_discardable(&mut self, name: &StringRef) -> bool {
        do_unsafe!(mlirOperationRemoveDiscardableAttributeByName(
            *self.get_mut(),
            *name.get()
        ))
    }

    pub fn remove_from_parent(&mut self) -> () {
        do_unsafe!(mlirOperationRemoveFromParent(*self.get_mut()))
    }

    pub fn replace_all_symbol_uses(
        &mut self,
        sym_old: &StringRef,
        sym_new: &StringRef,
    ) -> LogicalResult {
        LogicalResult::from(do_unsafe!(mlirSymbolTableReplaceAllSymbolUses(
            *sym_old.get(),
            *sym_new.get(),
            *self.get_mut(),
        )))
    }

    pub fn set_attribute_discardable(&mut self, name: &StringRef, attr: &Attribute) -> () {
        do_unsafe!(mlirOperationSetDiscardableAttributeByName(
            *self.get_mut(),
            *name.get(),
            *attr.get(),
        ))
    }

    pub fn set_attribute_inherent(&mut self, name: &StringRef, attr: &Attribute) -> () {
        do_unsafe!(mlirOperationSetInherentAttributeByName(
            *self.get_mut(),
            *name.get(),
            *attr.get()
        ))
    }

    pub fn set_named_attribute_discardable(&mut self, attr: &Named) -> () {
        self.set_attribute_discardable(&attr.get_identifier().as_string(), &attr.as_attribute());
    }

    pub fn set_named_attribute_inherent(&mut self, attr: &Named) -> () {
        self.set_attribute_inherent(&attr.get_identifier().as_string(), &attr.as_attribute());
    }

    pub fn set_operand(&mut self, i: isize, value: &Value) -> () {
        do_unsafe!(mlirOperationSetOperand(*self.get_mut(), i, *value.get()))
    }

    pub fn set_operands(&mut self, values: &[Value]) -> () {
        let v: Vec<MlirValue> = values.iter().map(|v| *v.get()).collect();
        do_unsafe!(mlirOperationSetOperands(
            *self.get_mut(),
            values.len() as isize,
            v.as_ptr()
        ))
    }

    pub fn set_specialized_attribute_discardable(&mut self, attr: &impl IAttributeNamed) -> () {
        self.set_named_attribute_discardable(&attr.as_named_attribute());
    }

    pub fn set_specialized_attribute_inherent(&mut self, attr: &impl IAttributeNamed) -> () {
        self.set_named_attribute_inherent(&attr.as_named_attribute());
    }

    pub fn set_successor(&mut self, i: isize, block: &mut Block) -> () {
        do_unsafe!(mlirOperationSetSuccessor(
            *self.get_mut(),
            i,
            *block.get_mut()
        ))
    }

    pub fn verify(&self) -> bool {
        do_unsafe!(mlirOperationVerify(self.0))
    }
}

impl OperationState {
    pub fn new(name: &StringRef, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirOperationStateGet(*name.get(), *loc.get())))
    }

    pub fn add_attributes(&mut self, attributes: &[Named]) -> () {
        let a: Vec<MlirNamedAttribute> = attributes.iter().map(|a| *a.get()).collect();
        do_unsafe!(mlirOperationStateAddAttributes(
            self.get_mut(),
            a.len() as isize,
            a.as_ptr()
        ))
    }

    pub fn add_operands(&mut self, operands: &[Value]) -> () {
        let o: Vec<MlirValue> = operands.iter().map(|o| *o.get()).collect();
        do_unsafe!(mlirOperationStateAddOperands(
            self.get_mut(),
            o.len() as isize,
            o.as_ptr()
        ))
    }

    pub fn add_regions(&mut self, regions: &[Region]) -> () {
        let r: Vec<MlirRegion> = regions.iter().map(|r| *r.get()).collect();
        do_unsafe!(mlirOperationStateAddOwnedRegions(
            self.get_mut(),
            r.len() as isize,
            r.as_ptr()
        ))
    }

    pub fn add_results(&mut self, types: &[Type]) -> () {
        let t: Vec<MlirType> = types.iter().map(|t| *t.get()).collect();
        do_unsafe!(mlirOperationStateAddResults(
            self.get_mut(),
            t.len() as isize,
            t.as_ptr()
        ))
    }

    pub fn add_successors(&mut self, successors: &[Block]) -> () {
        let b: Vec<MlirBlock> = successors.iter().map(|b| *b.get()).collect();
        do_unsafe!(mlirOperationStateAddSuccessors(
            self.get_mut(),
            b.len() as isize,
            b.as_ptr()
        ))
    }

    pub fn create_operation(&mut self) -> Operation {
        Operation::new(self)
    }

    pub fn get(&self) -> &MlirOperationState {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirOperationState {
        &mut self.0
    }

    pub fn set_infer_result_type(&mut self) -> () {
        do_unsafe!(mlirOperationStateEnableResultTypeInference(self.get_mut()))
    }
}

impl OpOperand {
    pub fn as_value(&self) -> Value {
        Value::from(do_unsafe!(mlirOpOperandGetValue(self.0)))
    }

    pub fn get(&self) -> &MlirOpOperand {
        &self.0
    }

    pub fn get_index(&self) -> usize {
        do_unsafe!(mlirOpOperandGetOperandNumber(self.0)) as usize
    }

    pub fn get_mut(&mut self) -> &mut MlirOpOperand {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        do_unsafe!(mlirOpOperandIsNull(self.0))
    }
}

impl Pass {
    pub fn get(&self) -> &MlirPass {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirPass {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn register_all_passes() -> () {
        do_unsafe!(mlirRegisterAllPasses())
    }
}

impl Region {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirRegionCreate()))
    }

    pub fn append_block(&mut self, block: &mut Block) -> () {
        do_unsafe!(mlirRegionAppendOwnedBlock(
            *self.get_mut(),
            *block.get_mut()
        ));
        *self.num_blocks_mut() += 1;
    }

    pub fn get(&self) -> &MlirRegion {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirRegion {
        &mut self.0
    }

    pub fn insert_block(&mut self, block: &mut Block, i: usize) -> () {
        if i > self.num_blocks() {
            eprintln!("Block position '{}' out of bounds", i);
            exit(ExitCode::IRError);
        }
        do_unsafe!(mlirRegionInsertOwnedBlock(
            *self.get_mut(),
            i as isize,
            *block.get_mut()
        ));
        *self.num_blocks_mut() += 1;
    }

    pub fn insert_block_after(&mut self, anchor: &Block, block: &mut Block) -> () {
        do_unsafe!(mlirRegionInsertOwnedBlockAfter(
            *self.get_mut(),
            *anchor.get(),
            *block.get_mut()
        ));
        *self.num_blocks_mut() += 1;
    }

    pub fn insert_block_before(&mut self, anchor: &Block, block: &mut Block) -> () {
        do_unsafe!(mlirRegionInsertOwnedBlockBefore(
            *self.get_mut(),
            *anchor.get(),
            *block.get_mut()
        ));
        *self.num_blocks_mut() += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.num_blocks() == 0 && self.iter().next().is_none()
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn iter(&self) -> RegionIter {
        RegionIter(self, None)
    }

    fn __num_blocks(&self) -> usize {
        self.iter().fold(0, |acc, _b| acc + 1) as usize
    }

    pub fn num_blocks(&self) -> usize {
        self.1
    }

    pub fn num_blocks_mut(&mut self) -> &mut usize {
        &mut self.1
    }

    pub fn take_body(&mut self, other: &mut Region) -> () {
        let n = other.num_blocks();
        do_unsafe!(mlirRegionTakeBody(self.0, *other.get()));
        *self.num_blocks_mut() = n;
    }
}

impl Iterator for RegionIter<'_> {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        match self.1 {
            None => {
                if self.0.is_null() {
                    return None;
                }
                let block = Block::from(do_unsafe!(mlirRegionGetFirstBlock(*self.0.get())));
                if block.is_null() {
                    None
                } else {
                    self.1 = Some(block);
                    self.1.clone()
                }
            }
            Some(ref mut b) => match b.next() {
                None => {
                    self.1 = None;
                    None
                }
                Some(b_) => {
                    self.1 = Some(Block::from(b_));
                    self.1.clone()
                }
            },
        }
    }
}

impl Registry {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirDialectRegistryCreate()))
    }

    pub fn get(&self) -> &MlirDialectRegistry {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirDialectRegistry {
        &mut self.0
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn register_all_dialects(&mut self) -> () {
        do_unsafe!(mlirRegisterAllDialects(*self.get_mut()))
    }

    pub fn register_arith(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__arith__(),
            *self.get_mut()
        ))
    }

    pub fn register_func(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__func__(),
            *self.get_mut()
        ))
    }

    pub fn register_gpu(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__gpu__(),
            *self.get_mut()
        ))
    }

    pub fn register_linalg(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__linalg__(),
            *self.get_mut()
        ))
    }

    pub fn register_llvm(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__llvm__(),
            *self.get_mut()
        ))
    }

    pub fn register_memref(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__memref__(),
            *self.get_mut()
        ))
    }

    pub fn register_shape(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__shape__(),
            *self.get_mut()
        ))
    }

    pub fn register_spirv(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__spirv__(),
            *self.get_mut()
        ))
    }

    pub fn register_tensor(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__tensor__(),
            *self.get_mut()
        ))
    }

    pub fn register_vector(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(
            mlirGetDialectHandle__vector__(),
            *self.get_mut()
        ))
    }
}

impl ShapeImpl<Vec<i64>> {
    pub fn get(&self) -> &[i64] {
        self.0.as_slice()
    }

    pub fn get_mut(&mut self) -> &mut [i64] {
        self.0.as_mut()
    }

    pub fn transpose(&self) -> Self {
        Self::from(self.to_vec_transpose())
    }
}

impl StringBacked {
    pub fn new(s: String) -> Self {
        Self::from(s)
    }

    pub fn as_ptr(&self) -> *const c_char {
        self.0.data
    }

    pub fn as_ptr_mut(&mut self) -> *mut c_char {
        self.0.data.cast_mut()
    }

    pub fn as_string_ref(&self) -> StringRef {
        StringRef::from(self.0)
    }

    pub fn get(&self) -> &MlirStringRef {
        &self.0
    }

    /// Returns the (owned) backing string.
    pub fn get_string(&self) -> &CString {
        &self.1
    }

    pub fn get_mut(&mut self) -> &mut MlirStringRef {
        &mut self.0
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_null(&self) -> bool {
        self.get().data.is_null()
    }

    pub fn len(&self) -> usize {
        self.0.length
    }
}

impl StringCallback {
    pub fn new() -> Self {
        Self::from(Self::print_string as StringCallbackFn)
    }

    /// # Safety
    /// `data` may be dereferenced by the call back function.
    pub unsafe fn apply(&self, s: &StringRef, data: *mut c_void) -> () {
        if let Some(f) = *self.get() {
            do_unsafe!(f(*s.get(), data))
        }
    }

    pub fn get(&self) -> &MlirStringCallback {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirStringCallback {
        &mut self.0
    }

    pub fn is_none(&self) -> bool {
        self.get().is_none()
    }

    #[allow(clippy::unnecessary_cast)]
    unsafe extern "C" fn print_string(s: MlirStringRef, data: *mut c_void) -> () {
        let Some(state) = do_unsafe!(StringCallbackState::from_ptr(data).as_mut()) else {
            eprintln!("Failed to convert pointer to string callback state");
            exit(ExitCode::IRError);
        };
        let data = state.get_data_mut().as_mut_ptr();
        let offset = state.num_bytes_written();
        let pos_last = offset + s.length;
        if pos_last > STATE_BUFFER_DATA_LENGTH {
            eprintln!(
                "Index {} out of bounds for string callback state data buffer",
                pos_last
            );
            exit(ExitCode::IRError);
        }
        for i in 0..s.length {
            unsafe {
                let c = *s.data.wrapping_add(i) as u8;
                *data.wrapping_add(offset + i) = c;
            }
            *state.num_bytes_written_mut() += 1;
        }
    }
}

impl StringCallbackState {
    pub fn new() -> Self {
        Self {
            bytes_written: 0,
            data: [0; STATE_BUFFER_DATA_LENGTH],
        }
    }

    pub fn from_ptr(p: *mut c_void) -> *mut Self {
        p.cast::<Self>()
    }

    pub fn as_void_ptr(&self) -> *const c_void {
        ptr::from_ref(self) as *const c_void
    }

    pub fn as_void_mut_ptr(&mut self) -> *mut c_void {
        ptr::from_mut(self) as *mut c_void
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data
    }

    pub fn get_data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn num_bytes_written(&self) -> usize {
        self.bytes_written
    }

    pub fn num_bytes_written_mut(&mut self) -> &mut usize {
        &mut self.bytes_written
    }

    pub fn reset(&mut self) -> () {
        *self.num_bytes_written_mut() = 0;
        for i in 0..STATE_BUFFER_DATA_LENGTH {
            self.get_data_mut()[i] = 0
        }
    }
}

impl StringRef {
    pub fn as_ptr(&self) -> *const c_char {
        self.0.data
    }

    pub fn as_ptr_mut(&mut self) -> *mut c_char {
        self.0.data.cast_mut()
    }

    pub fn get(&self) -> &MlirStringRef {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirStringRef {
        &mut self.0
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_null(&self) -> bool {
        self.get().data.is_null()
    }

    pub fn len(&self) -> usize {
        self.0.length
    }
}

impl SymbolTable {
    pub fn new(op: &Operation) -> Self {
        Self::from(do_unsafe!(mlirSymbolTableCreate(*op.get())))
    }

    pub fn get(&self) -> &MlirSymbolTable {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirSymbolTable {
        &mut self.0
    }

    pub fn get_symbol_attribute_name() -> StringRef {
        StringRef::from(do_unsafe!(mlirSymbolTableGetSymbolAttributeName()))
    }

    pub fn get_visibility_attribute_name() -> StringRef {
        StringRef::from(do_unsafe!(mlirSymbolTableGetVisibilityAttributeName()))
    }

    pub fn erase(&mut self, op: &Operation) -> () {
        do_unsafe!(mlirSymbolTableErase(self.0, *op.get()))
    }

    /// Requires: Interface::Symbol
    pub fn insert(&mut self, op: &Operation) -> Attribute {
        Attribute::from(do_unsafe!(mlirSymbolTableInsert(
            *self.get_mut(),
            *op.get()
        )))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn lookup(&self, name: &StringRef) -> Operation {
        Operation::from(do_unsafe!(mlirSymbolTableLookup(self.0, *name.get())))
    }
}

impl Type {
    pub fn from_parse(context: &Context, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirTypeParseGet(*context.get(), *s.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirTypeDump(self.0))
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirTypeGetContext(self.0)))
    }

    pub fn get_dialect(&self) -> Dialect {
        Dialect::from(do_unsafe!(mlirTypeGetDialect(self.0)))
    }

    pub fn get_id(&self) -> TypeID {
        TypeID::from(do_unsafe!(mlirTypeGetTypeID(self.0)))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn is_bool(&self) -> bool {
        self.is_integer() && self.get_width().unwrap_or(0) == 1
    }

    pub fn is_complex(&self) -> bool {
        do_unsafe!(mlirTypeIsAComplex(self.0))
    }

    pub fn is_float(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat(self.0))
    }

    pub fn is_function(&self) -> bool {
        do_unsafe!(mlirTypeIsAFunction(self.0))
    }

    pub fn is_index(&self) -> bool {
        do_unsafe!(mlirTypeIsAIndex(self.0))
    }

    pub fn is_integer(&self) -> bool {
        do_unsafe!(mlirTypeIsAInteger(self.0))
    }

    pub fn is_memref(&self) -> bool {
        do_unsafe!(mlirTypeIsAMemRef(self.0))
    }

    pub fn is_none(&self) -> bool {
        do_unsafe!(mlirTypeIsANone(self.0))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn is_opaque(&self) -> bool {
        do_unsafe!(mlirTypeIsAOpaque(self.0))
    }

    pub fn is_ranked_tensor(&self) -> bool {
        do_unsafe!(mlirTypeIsARankedTensor(self.0))
    }

    pub fn is_shaped(&self) -> bool {
        do_unsafe!(mlirTypeIsAShaped(self.0))
    }

    pub fn is_tensor(&self) -> bool {
        do_unsafe!(mlirTypeIsATensor(self.0))
    }

    pub fn is_tuple(&self) -> bool {
        do_unsafe!(mlirTypeIsATuple(self.0))
    }

    /// Unit type is not exposed by the C API.
    /// However, we can use the indirectly constructed Unit type from the Unit attribute.
    pub fn is_unit(&self) -> bool {
        self.get_id() == Unit::get_type_id()
    }

    pub fn is_unranked_memref(&self) -> bool {
        do_unsafe!(mlirTypeIsAUnrankedMemRef(self.0))
    }

    pub fn is_unranked_tensor(&self) -> bool {
        do_unsafe!(mlirTypeIsAUnrankedTensor(self.0))
    }

    pub fn is_vector(&self) -> bool {
        do_unsafe!(mlirTypeIsAVector(self.0))
    }

    print_method!(mlirTypePrint);
}

impl TypeID {
    /// # Safety
    /// May dereference raw pointer input `p`
    pub unsafe fn from_ptr(p: *const c_void) -> Self {
        if p.align_offset(8) != 0 {
            eprintln!("Pointer for TypeID must be 8-byte aligned");
            exit(ExitCode::IRError);
        }
        Self::from(do_unsafe!(mlirTypeIDCreate(p)))
    }

    pub fn get(&self) -> &MlirTypeID {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut MlirTypeID {
        &mut self.0
    }

    pub fn hash(&self) -> usize {
        do_unsafe!(mlirTypeIDHashValue(self.0))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }
}

impl Value {
    pub fn new_null() -> Self {
        let v = MlirValue { ptr: ptr::null() };
        Self::from(v)
    }

    fn check_argument(&self) -> () {
        if !self.is_argument() {
            eprint!("Value is not a block arg: ");
            self.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
    }

    fn check_result(&self) -> () {
        if !self.is_result() {
            eprint!("Value is not a result: ");
            self.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirValueDump(self.0))
    }

    pub fn get(&self) -> &MlirValue {
        &self.0
    }

    pub fn get_arg_owner(&self) -> Block {
        self.check_argument();
        Block::from(do_unsafe!(mlirBlockArgumentGetOwner(self.0)))
    }

    pub fn get_arg_pos(&self) -> isize {
        self.check_argument();
        do_unsafe!(mlirBlockArgumentGetArgNumber(self.0))
    }

    pub fn get_mut(&mut self) -> &mut MlirValue {
        &mut self.0
    }

    pub fn get_result_owner(&self) -> Operation {
        self.check_result();
        Operation::from(do_unsafe!(mlirOpResultGetOwner(self.0)))
    }

    pub fn get_type(&self) -> Type {
        Type::from(do_unsafe!(mlirValueGetType(self.0)))
    }

    pub fn has_owner(&self) -> bool {
        self.is_result()
    }

    pub fn is_argument(&self) -> bool {
        do_unsafe!(mlirValueIsABlockArgument(self.0))
    }

    pub fn is_null(&self) -> bool {
        self.get().ptr.is_null()
    }

    pub fn is_result(&self) -> bool {
        do_unsafe!(mlirValueIsAOpResult(self.0))
    }

    pub fn iter(&self) -> ValueIter {
        ValueIter(self, None)
    }

    print_method!(mlirValuePrint);

    pub fn set_arg_type(&mut self, t: &Type) -> () {
        self.check_argument();
        do_unsafe!(mlirBlockArgumentSetType(self.0, *t.get()))
    }

    pub fn set_type(&mut self, t: &Type) -> () {
        do_unsafe!(mlirValueSetType(self.0, *t.get()))
    }

    pub fn replace(&mut self, value: &Value) -> () {
        do_unsafe!(mlirValueReplaceAllUsesOfWith(self.0, *value.get()))
    }
}

///////////////////////////////
//  Trait Implementation
///////////////////////////////

impl Default for Attribute {
    fn default() -> Self {
        Self::new()
    }
}

impl From<MlirAttribute> for Attribute {
    fn from(attr: MlirAttribute) -> Self {
        Self(attr)
    }
}

impl IAttribute for Attribute {
    fn as_attribute(&self) -> Attribute {
        self.clone()
    }

    fn get(&self) -> &MlirAttribute {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirAttribute {
        self.get_mut()
    }
}

impl cmp::PartialEq for Attribute {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirAttributeEqual(self.0, rhs.0))
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl Destroy for Block {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirBlockDestroy(self.0))
    }
}

impl From<MlirBlock> for Block {
    fn from(block: MlirBlock) -> Self {
        let mut b = Self(block, 0);
        *b.num_operations_mut() = b.__num_operations();
        b
    }
}

impl Iterator for Block {
    type Item = MlirBlock;

    fn next(&mut self) -> Option<Self::Item> {
        let block = do_unsafe!(mlirBlockGetNextInRegion(self.0));
        if block.ptr.is_null() {
            None
        } else {
            self.0 = block;
            Some(block)
        }
    }
}

impl cmp::PartialEq for Block {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirBlockEqual(self.0, rhs.0))
    }
}

impl Iterator for BlockIter<'_> {
    type Item = Operation;

    fn next(&mut self) -> Option<Self::Item> {
        match self.1 {
            None => {
                if self.0.is_null() {
                    return None;
                }
                let op = Operation::from(do_unsafe!(mlirBlockGetFirstOperation(*self.0.get())));
                if op.is_null() {
                    None
                } else {
                    self.1 = Some(op);
                    self.1.clone()
                }
            }
            Some(ref mut o) => match o.next() {
                None => {
                    self.1 = None;
                    None
                }
                Some(o_) => {
                    self.1 = Some(Operation::from(o_));
                    self.1.clone()
                }
            },
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Destroy for Context {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirContextDestroy(self.0))
    }
}

impl From<MlirContext> for Context {
    fn from(context: MlirContext) -> Self {
        Self(context)
    }
}

impl From<MlirDialect> for Dialect {
    fn from(dialect: MlirDialect) -> Self {
        Self(dialect)
    }
}

impl cmp::PartialEq for Dialect {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirDialectEqual(self.0, rhs.0))
    }
}

impl From<MlirIdentifier> for Identifier {
    fn from(id: MlirIdentifier) -> Self {
        Self(id)
    }
}

impl cmp::PartialEq for Identifier {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirIdentifierEqual(self.0, rhs.0))
    }
}

impl Default for Location {
    fn default() -> Self {
        Location::new_unknown(&Context::default())
    }
}

impl From<MlirLocation> for Location {
    fn from(loc: MlirLocation) -> Self {
        Self(loc)
    }
}

impl cmp::PartialEq for Location {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirLocationEqual(self.0, rhs.0))
    }
}

impl From<MlirLogicalResult> for LogicalResult {
    fn from(loc: MlirLogicalResult) -> Self {
        Self(loc)
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new(&Location::default())
    }
}

impl Destroy for Module {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirModuleDestroy(self.0))
    }
}

impl From<MlirModule> for Module {
    fn from(module: MlirModule) -> Self {
        Self(module)
    }
}

impl cmp::PartialEq for Module {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_operation() == rhs.as_operation()
    }
}

impl Clone for Operation {
    fn clone(&self) -> Operation {
        Operation::from(do_unsafe!(mlirOperationClone(self.0)))
    }
}

impl Destroy for Operation {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirOperationDestroy(self.0))
    }
}

impl From<MlirOperation> for Operation {
    fn from(op: MlirOperation) -> Self {
        Self(op)
    }
}

impl Iterator for Operation {
    type Item = MlirOperation;

    fn next(&mut self) -> Option<Self::Item> {
        let op = do_unsafe!(mlirOperationGetNextInBlock(self.0));
        if op.ptr.is_null() {
            None
        } else {
            self.0 = op;
            Some(op)
        }
    }
}

impl cmp::PartialEq for Operation {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirOperationEqual(self.0, rhs.0))
    }
}

impl Iterator for OperationIter<'_> {
    type Item = Region;

    fn next(&mut self) -> Option<Self::Item> {
        match self.1 {
            None => {
                if self.0.is_null() {
                    return None;
                }
                let region = Region::from(do_unsafe!(mlirOperationGetFirstRegion(*self.0.get())));
                if region.is_null() {
                    None
                } else {
                    self.1 = Some(region);
                    self.1.clone()
                }
            }
            Some(ref mut r) => match r.next() {
                None => {
                    self.1 = None;
                    None
                }
                Some(r_) => {
                    self.1 = Some(Region::from(r_));
                    self.1.clone()
                }
            },
        }
    }
}

impl From<MlirOperationState> for OperationState {
    fn from(state: MlirOperationState) -> Self {
        Self(state)
    }
}

impl From<MlirOpOperand> for OpOperand {
    fn from(op: MlirOpOperand) -> Self {
        Self(op)
    }
}

impl Iterator for OpOperand {
    type Item = MlirOpOperand;

    fn next(&mut self) -> Option<Self::Item> {
        let op = do_unsafe!(mlirOpOperandGetNextUse(self.0));
        if op.ptr.is_null() {
            None
        } else {
            self.0 = op;
            Some(op)
        }
    }
}

impl From<MlirPass> for Pass {
    fn from(pass: MlirPass) -> Self {
        Self(pass)
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Destroy for Region {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirRegionDestroy(self.0))
    }
}

impl From<MlirRegion> for Region {
    fn from(region: MlirRegion) -> Self {
        let mut r = Self(region, 0);
        *r.num_blocks_mut() = r.__num_blocks();
        r
    }
}

impl Iterator for Region {
    type Item = MlirRegion;

    fn next(&mut self) -> Option<Self::Item> {
        let region = do_unsafe!(mlirRegionGetNextInOperation(self.0));
        if region.ptr.is_null() {
            None
        } else {
            self.0 = region;
            Some(region)
        }
    }
}

impl cmp::PartialEq for Region {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirRegionEqual(self.0, rhs.0))
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Destroy for Registry {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirDialectRegistryDestroy(self.0))
    }
}

impl From<MlirDialectRegistry> for Registry {
    fn from(registry: MlirDialectRegistry) -> Self {
        Self(registry)
    }
}

impl cmp::PartialEq for dyn Shape {
    fn eq(&self, rhs: &Self) -> bool {
        self.unpack() == rhs.unpack()
    }
}

impl From<Vec<i64>> for ShapeImpl<Vec<i64>> {
    fn from(v: Vec<i64>) -> Self {
        Self(v)
    }
}

impl From<&Vec<i64>> for ShapeImpl<Vec<i64>> {
    fn from(v: &Vec<i64>) -> Self {
        Self(v.clone())
    }
}

impl Shape for ShapeImpl<Vec<i64>> {
    fn rank(&self) -> isize {
        self.get().len() as isize
    }

    fn get(&self, i: isize) -> i64 {
        if i < 0 || i >= self.rank() {
            eprintln!("Index out of bounds for shape implementation");
            exit(ExitCode::IRError);
        }
        *self.get().get(i as usize).unwrap()
    }
}

impl Default for StringBacked {
    fn default() -> Self {
        Self::from(String::default())
    }
}

impl From<&str> for StringBacked {
    fn from(s: &str) -> Self {
        match Self::from_str(s) {
            Ok(s_) => s_,
            Err(msg) => {
                eprintln!(
                    "Failed to create backed string from string '{}': {}",
                    s, msg
                );
                exit(ExitCode::IRError);
            }
        }
    }
}

impl From<MlirStringRef> for StringBacked {
    fn from(s: MlirStringRef) -> Self {
        Self::from(&s)
    }
}

impl From<&MlirStringRef> for StringBacked {
    fn from(s: &MlirStringRef) -> Self {
        let mut c_string = do_unsafe!(CString::from_raw(s.data.cast_mut() as *mut c_char));
        Self(*s, mem::take(&mut c_string))
    }
}

impl From<String> for StringBacked {
    fn from(s: String) -> Self {
        Self::from(&s)
    }
}

impl From<&String> for StringBacked {
    fn from(s: &String) -> Self {
        Self::from(s.as_str())
    }
}

impl FromStr for StringBacked {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut c_string = CString::new(s).expect("Conversion to CString");
        Ok(Self(
            do_unsafe!(mlirStringRefCreateFromCString(c_string.as_ptr())),
            mem::take(&mut c_string),
        ))
    }
}

impl cmp::PartialEq for StringBacked {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirStringRefEqual(self.0, rhs.0))
    }
}

impl Default for StringCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl From<MlirStringCallback> for StringCallback {
    fn from(callback: MlirStringCallback) -> Self {
        Self(callback)
    }
}

impl From<StringCallbackFn> for StringCallback {
    fn from(callback: StringCallbackFn) -> Self {
        Self::from(Some(callback))
    }
}

impl Default for StringCallbackState {
    fn default() -> Self {
        Self::new()
    }
}

impl From<MlirStringRef> for StringRef {
    fn from(s: MlirStringRef) -> Self {
        Self(s)
    }
}

impl cmp::PartialEq for StringRef {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirStringRefEqual(self.0, rhs.0))
    }
}

impl Destroy for SymbolTable {
    fn destroy(&mut self) -> () {
        do_unsafe!(mlirSymbolTableDestroy(self.0))
    }
}

impl From<MlirSymbolTable> for SymbolTable {
    fn from(table: MlirSymbolTable) -> Self {
        Self(table)
    }
}

impl From<MlirType> for Type {
    fn from(t: MlirType) -> Self {
        Self(t)
    }
}

impl GetWidth for Type {}

impl IsPromotableTo<Self> for Type {
    fn is_promotable_to(&self, other: &Self) -> bool {
        <dyn IType>::is_promotable_to(self, other)
    }
}

impl IType for Type {
    fn as_type(&self) -> Type {
        self.clone()
    }

    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl cmp::PartialEq for Type {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirTypeEqual(self.0, rhs.0))
    }
}

impl From<MlirTypeID> for TypeID {
    fn from(id: MlirTypeID) -> Self {
        Self(id)
    }
}

impl cmp::PartialEq for TypeID {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirTypeIDEqual(self.0, rhs.0))
    }
}

impl From<MlirValue> for Value {
    fn from(value: MlirValue) -> Self {
        Self(value)
    }
}

impl cmp::PartialEq for Value {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirValueEqual(self.0, rhs.0))
    }
}

impl Iterator for ValueIter<'_> {
    type Item = OpOperand;

    fn next(&mut self) -> Option<Self::Item> {
        match self.1 {
            None => {
                if self.0.is_null() {
                    return None;
                }
                let op = OpOperand::from(do_unsafe!(mlirValueGetFirstUse(*self.0.get())));
                if op.is_null() {
                    None
                } else {
                    self.1 = Some(op);
                    self.1.clone()
                }
            }
            Some(ref mut o) => match o.next() {
                None => {
                    self.1 = None;
                    None
                }
                Some(o_) => {
                    self.1 = Some(OpOperand::from(o_));
                    self.1.clone()
                }
            },
        }
    }
}

///////////////////////////////
//  Display
///////////////////////////////

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.as_operation().print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for ShapeImpl<Vec<i64>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.get()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )
    }
}

impl fmt::Display for StringBacked {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return Ok(());
        }
        let s = match self.get_string().to_str() {
            Ok(s) => s,
            Err(msg) => panic!("Failed to convert CString: {}", msg),
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for StringCallbackState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const MSG: &str = "Failed to convert to string from string callback state data";
        let Ok(mut string) = String::from_utf8(self.get_data().to_vec()) else {
            panic!("{}", MSG)
        };
        string.truncate(self.num_bytes_written());
        string.shrink_to_fit();
        write!(f, "{}", string)
    }
}

#[allow(clippy::unnecessary_cast)]
impl fmt::Display for StringRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return Ok(());
        }
        let mut v: Vec<u8> = Vec::new();
        for i in 0..self.len() {
            let p = do_unsafe!(self.as_ptr().add(i));
            if !p.is_null() {
                // The type of `*p` is `i8` on darwin and `u8` on linux.
                v.push(do_unsafe!(*p) as u8);
            }
        }
        let c_string = match CString::new(v) {
            Ok(s) => s,
            Err(msg) => panic!("Failed to create CString: {}", msg),
        };
        let s = match c_string.to_str() {
            Ok(s) => s,
            Err(msg) => panic!("Failed to convert CString: {}", msg),
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut state = StringCallbackState::new();
        self.print(&mut state);
        write!(f, "{}", state)
    }
}
