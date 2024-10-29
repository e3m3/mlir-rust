// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirAttributeDump;
use mlir::mlirAttributeGetContext;
use mlir::mlirAttributeGetDialect;
use mlir::mlirAttributeGetNull;
use mlir::mlirAttributeGetType;
use mlir::mlirAttributeGetTypeID;
use mlir::mlirAttributeIsAAffineMap;
use mlir::mlirAttributeIsAArray;
use mlir::mlirAttributeIsABool;
use mlir::mlirAttributeIsADenseBoolArray;
use mlir::mlirAttributeIsADenseF32Array;
use mlir::mlirAttributeIsADenseF64Array;
use mlir::mlirAttributeIsADenseI8Array;
use mlir::mlirAttributeIsADenseI16Array;
use mlir::mlirAttributeIsADenseI32Array;
use mlir::mlirAttributeIsADenseI64Array;
use mlir::mlirAttributeIsADenseElements;
use mlir::mlirAttributeIsADenseFPElements;
use mlir::mlirAttributeIsADenseIntElements;
use mlir::mlirAttributeIsADenseResourceElements;
use mlir::mlirAttributeIsADictionary;
use mlir::mlirAttributeIsAElements;
use mlir::mlirAttributeIsAFlatSymbolRef;
use mlir::mlirAttributeIsAFloat;
use mlir::mlirAttributeIsAInteger;
use mlir::mlirAttributeIsAIntegerSet;
use mlir::mlirAttributeIsALocation;
use mlir::mlirAttributeIsAOpaque;
use mlir::mlirAttributeIsASparseElements;
use mlir::mlirAttributeIsAStridedLayout;
use mlir::mlirAttributeIsAString;
use mlir::mlirAttributeIsASymbolRef;
use mlir::mlirAttributeIsAType;
use mlir::mlirAttributeIsAUnit;
use mlir::mlirAttributeParseGet;
use mlir::mlirBlockAddArgument;
use mlir::mlirBlockCreate;
use mlir::mlirBlockDestroy;
use mlir::mlirBlockEqual;
use mlir::mlirBlockGetArgument;
use mlir::mlirBlockGetFirstOperation;
use mlir::mlirBlockGetNextInRegion;
use mlir::mlirBlockGetNumArguments;
use mlir::mlirBlockGetParentRegion;
use mlir::mlirBlockGetTerminator;
use mlir::mlirContextCreate;
use mlir::mlirContextCreateWithRegistry;
use mlir::mlirContextDestroy;
use mlir::mlirContextGetOrLoadDialect;
use mlir::mlirDialectEqual;
use mlir::mlirDialectGetContext;
use mlir::mlirDialectGetNamespace;
use mlir::mlirDialectHandleInsertDialect;
use mlir::mlirDialectHandleLoadDialect;
use mlir::mlirDialectRegistryCreate;
use mlir::mlirDialectRegistryDestroy;
use mlir::mlirDisctinctAttrCreate;
use mlir::mlirGetDialectHandle__arith__;
use mlir::mlirGetDialectHandle__gpu__;
use mlir::mlirGetDialectHandle__linalg__;
use mlir::mlirGetDialectHandle__llvm__;
use mlir::mlirGetDialectHandle__shape__;
use mlir::mlirGetDialectHandle__spirv__;
use mlir::mlirGetDialectHandle__tensor__;
use mlir::mlirGetDialectHandle__vector__;
use mlir::mlirIdentifierEqual;
use mlir::mlirIdentifierGet;
use mlir::mlirIdentifierGetContext;
use mlir::mlirIdentifierStr;
use mlir::mlirIntegerSetDump;
use mlir::mlirIntegerSetEqual;
use mlir::mlirIntegerSetEmptyGet;
use mlir::mlirIntegerSetGet;
use mlir::mlirIntegerSetGetContext;
use mlir::mlirLocationCallSiteGet;
use mlir::mlirLocationEqual;
use mlir::mlirLocationFileLineColGet;
use mlir::mlirLocationFromAttribute;
use mlir::mlirLocationGetAttribute;
use mlir::mlirLocationUnknownGet;
use mlir::mlirModuleCreateEmpty;
use mlir::mlirModuleCreateParse;
use mlir::mlirModuleDestroy;
use mlir::mlirModuleGetBody;
use mlir::mlirModuleGetOperation;
use mlir::mlirOperationClone;
use mlir::mlirOperationCreateParse;
use mlir::mlirOperationDump;
use mlir::mlirOperationGetDiscardableAttributeByName;
use mlir::mlirOperationGetFirstRegion;
use mlir::mlirOperationGetInherentAttributeByName;
use mlir::mlirOperationGetName;
use mlir::mlirOperationGetNextInBlock;
use mlir::mlirOperationGetNumRegions;
use mlir::mlirOperationGetOperand;
use mlir::mlirOperationGetRegion;
use mlir::mlirOperationGetResult;
use mlir::mlirOperationMoveAfter;
use mlir::mlirOperationMoveBefore;
use mlir::mlirOperationSetDiscardableAttributeByName;
use mlir::mlirOperationSetInherentAttributeByName;
use mlir::mlirOperationSetOperand;
use mlir::mlirOperationSetOperands;
use mlir::mlirOperationVerify;
use mlir::mlirOpOperandGetNextUse;
use mlir::mlirOpOperandGetOperandNumber;
use mlir::mlirOpOperandGetValue;
use mlir::mlirOpOperandIsNull;
use mlir::mlirOpResultGetOwner;
use mlir::mlirRegionCreate;
use mlir::mlirRegionDestroy;
use mlir::mlirRegionEqual;
use mlir::mlirRegionGetFirstBlock;
use mlir::mlirRegionGetNextInOperation;
use mlir::mlirRegisterAllPasses;
use mlir::mlirStringRefCreateFromCString;
use mlir::mlirSymbolTableCreate;
use mlir::mlirSymbolTableDestroy;
use mlir::mlirSymbolTableErase;
use mlir::mlirSymbolTableInsert;
use mlir::mlirSymbolTableLookup;
use mlir::mlirValueDump;
use mlir::mlirValueEqual;
use mlir::mlirValueGetFirstUse;
use mlir::mlirValueGetType;
use mlir::mlirValueIsABlockArgument;
use mlir::mlirValueIsAOpResult;
use mlir::mlirValueReplaceAllUsesOfWith;
use mlir::mlirValueSetType;
use mlir::mlirTypeDump;
use mlir::mlirTypeGetContext;
use mlir::mlirTypeGetDialect;
use mlir::mlirTypeGetTypeID;
use mlir::mlirTypeIsAComplex;
use mlir::mlirTypeIsAFloat;
use mlir::mlirTypeIsAFunction;
use mlir::mlirTypeIsAInteger;
use mlir::mlirTypeIsAMemRef;
use mlir::mlirTypeIsANone;
use mlir::mlirTypeIsAOpaque;
use mlir::mlirTypeIsARankedTensor;
use mlir::mlirTypeIsAShaped;
use mlir::mlirTypeIsATensor;
use mlir::mlirTypeIsATuple;
use mlir::mlirTypeIsAVector;
use mlir::mlirTypeIsAUnrankedMemRef;
use mlir::mlirTypeIsAUnrankedTensor;
use mlir::mlirTypeParseGet;
use mlir::mlirTypeIDCreate;
use mlir::mlirTypeIDEqual;
use mlir::mlirTypeIDHashValue;
use mlir::MlirAffineExpr;
use mlir::MlirAttribute;
use mlir::MlirBlock;
use mlir::MlirContext;
use mlir::MlirDialect;
use mlir::MlirDialectRegistry;
use mlir::MlirIdentifier;
use mlir::MlirIntegerSet;
use mlir::MlirLocation;
use mlir::MlirLogicalResult;
use mlir::MlirModule;
use mlir::MlirPass;
use mlir::MlirOperation;
use mlir::MlirOpOperand;
use mlir::MlirRegion;
use mlir::MlirStringCallback;
use mlir::MlirStringRef;
use mlir::MlirSymbolTable;
use mlir::MlirType;
use mlir::MlirTypeID;
use mlir::MlirValue;

use std::cmp;
use std::ffi::c_char;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::ffi::CString;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::do_unsafe;
use crate::exit_code;
use crate::types;

use attributes::IRAttribute;
use dialects::affine;
use exit_code::exit;
use exit_code::ExitCode;
use types::IRType;

pub trait Shape {
    fn rank(&self) -> usize;
    fn get(&self) -> &[u64];
}

#[derive(Clone)]
pub struct Attribute(MlirAttribute);

#[derive(Clone)]
pub struct Block(MlirBlock);

#[derive(Clone)]
pub struct Context(MlirContext);

#[derive(Clone)]
pub struct Dialect(MlirDialect);

#[derive(Clone)]
pub struct Identifier(MlirIdentifier);

#[derive(Clone)]
pub struct IntegerSet(MlirIntegerSet);

#[derive(Clone)]
pub struct Location(MlirLocation);

#[derive(Clone)]
pub struct LogicalResult(MlirLogicalResult);

#[derive(Clone)]
pub struct Module(MlirModule);

#[derive(Clone)]
pub struct Pass(MlirPass);

pub struct Operation(MlirOperation);

#[derive(Clone)]
pub struct OpOperand(MlirOpOperand);

#[derive(Clone)]
pub struct Region(MlirRegion);

#[derive(Clone)]
pub struct Registry(MlirDialectRegistry);

#[derive(Clone)]
pub struct StringCallback(MlirStringCallback);
pub type StringCallbackFn = unsafe extern "C" fn(MlirStringRef, *mut c_void);

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

impl Attribute {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirAttributeGetNull()))
    }

    /// This is the only mention of DisctinctAttr.
    /// Also, why is it mispelled?
    pub fn new_disctinct(attr: &Attribute) -> Self {
        Self::from(do_unsafe!(mlirDisctinctAttrCreate(*attr.get())))
    }

    pub fn from(attr: MlirAttribute) -> Self {
        Attribute(attr)
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

    pub fn is_integer(&self) -> bool {
        do_unsafe!(mlirAttributeIsAInteger(self.0))
    }

    pub fn is_integer_set(&self) -> bool {
        do_unsafe!(mlirAttributeIsAIntegerSet(self.0))
    }

    pub fn is_location(&self) -> bool {
        do_unsafe!(mlirAttributeIsALocation(self.0))
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

    pub fn to_location(&self) -> Location {
        Location::from(do_unsafe!(mlirLocationFromAttribute(self.0)))
    }
}

impl Default for Attribute {
    fn default() -> Self {
        Self::new()
    }
}

impl IRAttribute for Attribute {
    fn as_attribute(&self) -> Attribute {
        self.clone()
    }

    fn get(&self) -> &MlirAttribute {
        &self.0
    }
}

impl Block {
    pub fn new(num_args: isize, args: &[Type], locs: &[Location]) -> Self {
        assert!(num_args == args.len() as isize);
        assert!(num_args == locs.len() as isize);
        let args_raw: Vec<MlirType> = args.iter().map(|a| *a.get()).collect();
        let locs_raw: Vec<MlirLocation> = locs.iter().map(|m| *m.get()).collect();
        Self::from(do_unsafe!(mlirBlockCreate(num_args, args_raw.as_ptr(), locs_raw.as_ptr())))
    }

    pub fn from(block: MlirBlock) -> Self {
        Block(block)
    }

    pub fn add_arg(&mut self, t: &Type, loc: &Location) -> Value {
        Value::from(do_unsafe!(mlirBlockAddArgument(self.0, *t.get(), *loc.get())))
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

    pub fn get_parent(&self) -> Region {
        Region::from(do_unsafe!(mlirBlockGetParentRegion(self.0)))
    }

    pub fn get_terminator(&self) -> Operation {
        Operation::from(do_unsafe!(mlirBlockGetTerminator(self.0)))
    }

    pub fn iter(&self) -> Operation {
        Operation::from(do_unsafe!(mlirBlockGetFirstOperation(self.0)))
    }

    pub fn num_args(&self) -> isize {
        do_unsafe!(mlirBlockGetNumArguments(self.0))
    }
}

impl Context {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirContextCreate()))
    }

    pub fn from(context: MlirContext) -> Self {
        Context(context)
    }

    pub fn from_registry(registry: &Registry) -> Self {
        Self::from(do_unsafe!(mlirContextCreateWithRegistry(*registry.get(), false)))
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

    pub fn get_unknown_location(&self) -> Location {
        Location::unknown_from(self)
    }

    /// Load a registered dialect with name
    pub fn load_dialect(&self, name: &str) -> Option<Dialect> {
        let string = StringRef::from_str(name).unwrap();
        let dialect = do_unsafe!(mlirContextGetOrLoadDialect(self.0, *string.get()));
        if dialect.ptr.is_null() {
            None 
        } else {
            Some(Dialect::from(dialect))
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Context {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirContextDestroy(self.0))
    }
}

impl Drop for Block {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirBlockDestroy(self.0))
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

impl Dialect {
    pub fn from(dialect: MlirDialect) -> Self {
        Dialect(dialect)
    }

    pub fn get(&self) -> &MlirDialect {
        &self.0
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirDialectGetContext(self.0)))
    }

    pub fn get_namespace(&self) -> StringRef {
        StringRef::from(do_unsafe!(mlirDialectGetNamespace(self.0)))
    }
}

impl cmp::PartialEq for Dialect {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirDialectEqual(self.0, rhs.0))
    }
}

impl Identifier {
    pub fn from(id: MlirIdentifier) -> Self {
        Identifier(id)
    }

    pub fn from_string(context: &Context, s: &StringRef) -> Self {
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
}

impl cmp::PartialEq for Identifier {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirIdentifierEqual(self.0, rhs.0))
    }
}

impl IntegerSet {
    pub fn new(
        context: &Context,
        num_dims: isize,
        num_syms: isize,
        constraints: &[affine::Expr],
        flags: &[bool],
    ) -> Self {
        let c_len = constraints.len();
        let f_len = flags.len();
        if c_len != f_len {
            eprintln!("Mismatched constraints ('{}') and flags ('{}') sizes", c_len, f_len);
            exit(ExitCode::IRError);
        }
        let c: Vec<MlirAffineExpr> = constraints.iter().map(|e| *e.get()).collect();
        Self::from(do_unsafe!(mlirIntegerSetGet(
            *context.get(),
            num_dims,
            num_syms,
            c_len as isize,
            c.as_ptr(),
            flags.as_ptr(),
        )))
    }

    pub fn new_empty(context: &Context, num_dims: isize, num_syms: isize) -> Self {
        Self::from(do_unsafe!(mlirIntegerSetEmptyGet(*context.get(), num_dims, num_syms)))
    }

    pub fn from(set: MlirIntegerSet) -> Self {
        IntegerSet(set)
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirIntegerSetDump(self.0))
    }

    pub fn get_context(&self) -> Context {
        Context::from(do_unsafe!(mlirIntegerSetGetContext(self.0)))
    }
}

impl cmp::PartialEq for IntegerSet {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirIntegerSetEqual(self.0, rhs.0))
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

    pub fn from(loc: MlirLocation) -> Self {
        Location(loc)
    }

    pub fn from_call_site(callee: &Location, caller: &Location) -> Self {
        Self::from(do_unsafe!(mlirLocationCallSiteGet(*callee.get(), *caller.get())))
    }

    pub fn get(&self) -> &MlirLocation {
        &self.0
    }

    pub fn get_attribute(&self) -> Attribute {
        Attribute::from(do_unsafe!(mlirLocationGetAttribute(self.0)))
    }

    pub fn unknown_from(context: &Context) -> Self {
        Location::from(do_unsafe!(mlirLocationUnknownGet(*context.get())))
    }
}

impl Default for Location {
    fn default() -> Self {
        Location::unknown_from(&Context::default())
    }
}

impl cmp::PartialEq for Location {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirLocationEqual(self.0, rhs.0))
    }
}

impl LogicalResult {
    pub fn from(loc: MlirLogicalResult) -> Self {
        LogicalResult(loc)
    }

    pub fn get(&self) -> &MlirLogicalResult {
        &self.0
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

    pub fn from(module: MlirModule) -> Self {
        Module(module)
    }

    pub fn from_parse(context: &Context, string: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirModuleCreateParse(*context.get(), *string.get())))
    }

    pub fn get_body(&self) -> Block {
        Block::from(do_unsafe!(mlirModuleGetBody(self.0)))
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new(&Location::default())
    }
}

impl Drop for Module {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirModuleDestroy(self.0))
    }
}

impl Pass {
    pub fn from(pass: MlirPass) -> Self {
        Pass(pass)
    }

    pub fn get(&self) -> &MlirPass {
        &self.0
    }

    pub fn register_all_passes() -> () {
        do_unsafe!(mlirRegisterAllPasses())
    }
}

impl Operation {
    pub fn from(op: MlirOperation) -> Self {
        Operation(op)
    }

    pub fn from_parse(context: &Context, op: &StringRef, src_name: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirOperationCreateParse(*context.get(), *op.get(), *src_name.get())))
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirOperationDump(self.0))
    }

    pub fn get(&self) -> &MlirOperation {
        &self.0
    }

    pub fn get_attribute_discardable(&self, name: &StringRef) -> Attribute {
        Attribute::from(do_unsafe!(mlirOperationGetDiscardableAttributeByName(self.0, *name.get())))
    }

    pub fn get_attribute_inherent(&self, name: &StringRef) -> Attribute {
        Attribute::from(do_unsafe!(mlirOperationGetInherentAttributeByName(self.0, *name.get())))
    }

    pub fn get_name(&self) -> Identifier {
        Identifier::from(do_unsafe!(mlirOperationGetName(self.0)))
    }

    pub fn get_operand(&self, i: isize) -> Value {
        Value::from(do_unsafe!(mlirOperationGetOperand(self.0, i)))
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

    pub fn get_symbol_table(&self) -> Option<SymbolTable> {
        let table = do_unsafe!(mlirSymbolTableCreate(self.0));
        if table.ptr.is_null() {
            None
        } else {
            Some(SymbolTable::from(table))
        }
    }

    pub fn insert_after(&mut self, other: &Self) -> () {
        do_unsafe!(mlirOperationMoveAfter(self.0, other.0))
    }

    pub fn insert_before(&mut self, other: &Self) -> () {
        do_unsafe!(mlirOperationMoveBefore(self.0, other.0))
    }

    pub fn iter(&self) -> Region {
        Region::from(do_unsafe!(mlirOperationGetFirstRegion(self.0)))
    }

    pub fn num_regions(&self) -> isize {
        do_unsafe!(mlirOperationGetNumRegions(self.0))
    }

    pub fn set_attribute_discardable(&mut self, name: &StringRef, attr: &Attribute) -> () {
        do_unsafe!(mlirOperationSetDiscardableAttributeByName(self.0, *name.get(), *attr.get()))
    }

    pub fn set_attribute_inherent(&mut self, name: &StringRef, attr: &Attribute) -> () {
        do_unsafe!(mlirOperationSetInherentAttributeByName(self.0, *name.get(), *attr.get()))
    }

    pub fn set_operand(&mut self, i: isize, value: &Value) -> () {
        do_unsafe!(mlirOperationSetOperand(self.0, i, *value.get()))
    }

    pub fn set_operands(&mut self, values: &[Value]) -> () {
        let v: Vec<MlirValue> = values.iter().map(|v| *v.get()).collect();
        do_unsafe!(mlirOperationSetOperands(self.0, values.len() as isize, v.as_ptr()))
    }

    pub fn verify(&self) -> bool {
        do_unsafe!(mlirOperationVerify(self.0))
    }
}

impl Clone for Operation {
    fn clone(&self) -> Operation {
        Operation::from(do_unsafe!(mlirOperationClone(self.0)))
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

impl OpOperand {
    pub fn from(op: MlirOpOperand) -> Self {
        OpOperand(op)
    }

    pub fn as_value(&self) -> Value {
        Value::from(do_unsafe!(mlirOpOperandGetValue(self.0)))
    }

    pub fn get(&self) -> &MlirOpOperand {
        &self.0
    }

    pub fn get_index(&self) -> usize {
        do_unsafe!(mlirOpOperandGetOperandNumber(self.0)) as usize
    }

    pub fn is_null(&self) -> bool {
        do_unsafe!(mlirOpOperandIsNull(self.0))
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

impl Region {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirRegionCreate()))
    }

    pub fn from(region: MlirRegion) -> Region {
        Region(region)
    }

    pub fn iter(&self) -> Block {
        Block::from(do_unsafe!(mlirRegionGetFirstBlock(self.0)))
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirRegionDestroy(self.0))
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

impl Registry {
    pub fn new() -> Self {
        Self::from(do_unsafe!(mlirDialectRegistryCreate()))
    }

    pub fn from(registry: MlirDialectRegistry) -> Self {
        Registry(registry)
    }

    pub fn get(&self) -> &MlirDialectRegistry {
        &self.0
    }

    pub fn register_arith(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__arith__(), self.0))
    }

    pub fn register_gpu(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__gpu__(), self.0))
    }

    pub fn register_linalg(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__linalg__(), self.0))
    }

    pub fn register_llvm(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__llvm__(), self.0))
    }

    pub fn register_shape(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__shape__(), self.0))
    }

    pub fn register_spirv(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__spirv__(), self.0))
    }

    pub fn register_tensor(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__tensor__(), self.0))
    }

    pub fn register_vector(&mut self) -> () {
        do_unsafe!(mlirDialectHandleInsertDialect(mlirGetDialectHandle__vector__(), self.0))
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Registry {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirDialectRegistryDestroy(self.0))
    }
}

impl StringCallback {
    pub fn from(callback: MlirStringCallback) -> Self {
        StringCallback(callback)
    }

    pub fn from_fn(callback: StringCallbackFn) -> Self {
        Self::from(Some(callback))
    }

    /// # Safety
    /// `data` may be dereferenced by the call back function.
    pub unsafe fn apply(&self, s: &StringRef, data: *mut c_void) -> () {
        if self.0.is_some() {
            do_unsafe!(self.0.unwrap()(*s.get(), data))
        }
    }

    pub fn get(&self) -> &MlirStringCallback {
        &self.0
    }
}

impl StringRef {
    pub fn from(s: MlirStringRef) -> Self {
        StringRef(s)
    }

    pub fn as_ptr(&self) -> *const c_char {
        self.0.data
    }

    pub fn get(&self) -> &MlirStringRef {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.0.length
    }
}

impl FromStr for StringRef {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c_string = CString::new(format!("{}\0", s)).expect("Conversion to CString");
        let s = do_unsafe!(mlirStringRefCreateFromCString(c_string.as_ptr() as *const c_char));
        Ok(Self::from(s))
    }
}

impl SymbolTable {
    pub fn from(table: MlirSymbolTable) -> Self {
        SymbolTable(table)
    }

    pub fn get(&self) -> &MlirSymbolTable {
        &self.0
    }

    pub fn erase(&mut self, op: &Operation) -> () {
        do_unsafe!(mlirSymbolTableErase(self.0, *op.get()))
    }

    pub fn insert(&mut self, op: &Operation) -> Attribute {
        Attribute::from(do_unsafe!(mlirSymbolTableInsert(self.0, *op.get())))
    }

    pub fn lookup(&self, name: &StringRef) -> Operation {
        Operation::from(do_unsafe!(mlirSymbolTableLookup(self.0, *name.get())))
    }
}

impl Drop for SymbolTable {
    fn drop(&mut self) -> () {
        do_unsafe!(mlirSymbolTableDestroy(self.0))
    }
}

impl Type {
    pub fn from(t: MlirType) -> Self {
        Type(t)
    }

    pub fn from_parse(context: &Context, s: &StringRef) -> Self {
        Self::from(do_unsafe!(mlirTypeParseGet(*context.get(), *s.get())))
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirTypeDump(self.0))
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

    pub fn is_complex(&self) -> bool {
        do_unsafe!(mlirTypeIsAComplex(self.0))
    }

    pub fn is_float(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat(self.0))
    }

    pub fn is_function(&self) -> bool {
        do_unsafe!(mlirTypeIsAFunction(self.0))
    }

    pub fn is_integer(&self) -> bool {
        do_unsafe!(mlirTypeIsAInteger(self.0))
    }

    pub fn is_mem_ref(&self) -> bool {
        do_unsafe!(mlirTypeIsAMemRef(self.0))
    }

    pub fn is_none(&self) -> bool {
        do_unsafe!(mlirTypeIsANone(self.0))
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

    pub fn is_vector(&self) -> bool {
        do_unsafe!(mlirTypeIsAVector(self.0))
    }

    pub fn is_unranked_mem_ref(&self) -> bool {
        do_unsafe!(mlirTypeIsAUnrankedMemRef(self.0))
    }

    pub fn is_unranked_tensor(&self) -> bool {
        do_unsafe!(mlirTypeIsAUnrankedTensor(self.0))
    }
}

impl IRType for Type {
    fn as_type(&self) -> Type {
        self.clone()
    }

    fn get(&self) -> &MlirType {
        &self.0
    }
}

impl TypeID {
    pub fn from(id: MlirTypeID) -> Self {
        TypeID(id)
    }

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

    pub fn hash(&self) -> usize {
        do_unsafe!(mlirTypeIDHashValue(self.0))
    }
}

impl cmp::PartialEq for TypeID {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirTypeIDEqual(self.0, rhs.0))
    }
}

impl Value {
    pub fn from(value: MlirValue) -> Self {
        Value(value)
    }

    pub fn dump(&self) -> () {
        do_unsafe!(mlirValueDump(self.0))
    }

    pub fn get(&self) -> &MlirValue {
        &self.0
    }

    pub fn get_owner(&self) -> Operation {
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

    pub fn is_result(&self) -> bool {
        do_unsafe!(mlirValueIsAOpResult(self.0))
    }

    pub fn iter(&self) -> OpOperand {
        OpOperand::from(do_unsafe!(mlirValueGetFirstUse(self.0)))
    }

    pub fn set_type(&mut self, t: &Type) -> () {
        do_unsafe!(mlirValueSetType(self.0, *t.get()))
    }

    pub fn replace(&mut self, value: &Value) -> () {
        do_unsafe!(mlirValueReplaceAllUsesOfWith(self.0, *value.get()))
    }
}

impl cmp::PartialEq for Value {
    fn eq(&self, rhs: &Self) -> bool {
        do_unsafe!(mlirValueEqual(self.0, rhs.0))
    }
}
