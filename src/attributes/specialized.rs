// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirAttribute;

use std::cmp;
use std::ffi::c_uint;
use std::fmt;
use std::str::FromStr;

use crate::attributes;
use crate::dialects;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::IAttribute;
use attributes::IAttributeNamed;
use attributes::array::Array;
use attributes::bool::Bool as BoolAttr;
use attributes::dense_array::DenseArray;
use attributes::dense_array::Layout as DenseArrayLayout;
use attributes::dictionary::Dictionary;
use attributes::elements::Elements;
use attributes::float::Float as FloatAttr;
use attributes::index::Index as IndexAttr;
use attributes::integer::Integer as IntegerAttr;
use attributes::integer_set::IntegerSet;
use attributes::opaque::Opaque;
use attributes::strided_layout::StridedLayout;
use attributes::string::String as StringAttr;
use attributes::symbol_ref::SymbolRef;
use attributes::r#type::Type as TypeAttr;
use attributes::unit::Unit;
use dialects::affine::Map as AffineMap;
use dialects::affine::Set as AffineSet;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Attribute;
use ir::Context;
use ir::StringBacked;
use ir::StringRef;
use ir::Type;
use types::IType;
use types::function::Function;
use types::integer::Integer as IntegerType;
use types::memref::MemRef;

///////////////////////////////
//  Support
///////////////////////////////

pub struct CustomAttributeData {
    name: String,
    namespace: String,
    data: Vec<String>,
}

///////////////////////////////
//  Specialized Traits
///////////////////////////////

pub trait NamedAffineMap: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(map: AffineMap) -> Self {
        Self::from(*map.as_attribute().get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_affine_map() {
            eprintln!("Expected affine map attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_affine_map(&self) -> AffineMap {
        AffineMap::from_attribute(&Attribute::from(*self.get()))
    }
}

pub trait NamedAffineSet: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(set: AffineSet) -> Self {
        Self::from(*IntegerSet::new(set).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_integer_set() {
            eprintln!("Expected integer set attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }
}

pub trait NamedArrayOfAffineMaps: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, elements: &[AffineMap]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of affine map attributes");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_affine_map()) {
            eprintln!("Expected array of affine map attributes");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_affine_maps(&self) -> Vec<AffineMap> {
        let args = self.as_array();
        (0..args.num_elements())
            .map(|i| AffineMap::from(*args.get_element(i).get()))
            .collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfBools: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, elements: &[BoolAttr]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of bool attributes");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_bool()) {
            eprintln!("Expected array of bool attributes");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_bools(&self) -> Vec<BoolAttr> {
        let args = self.as_array();
        (0..args.num_elements())
            .map(|i| BoolAttr::from(*args.get_element(i).get()))
            .collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfDictionaries: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, elements: &[Dictionary]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_dictionary()) {
            eprintln!("Expected array of dictionary attributes");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_dictionaries(&self) -> Vec<Dictionary> {
        let args = self.as_array();
        (0..args.num_elements())
            .map(|i| Dictionary::from(*args.get_element(i).get()))
            .collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfIntegers: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, elements: &[IntegerAttr]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of integer attributes");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_integer()) {
            eprintln!("Expected array of integer attributes");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_integers(&self) -> Vec<IntegerAttr> {
        let args = self.as_array();
        (0..args.num_elements())
            .map(|i| IntegerAttr::from(*args.get_element(i).get()))
            .collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfIntegerArrays: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, values: &[Array], width: usize) -> Self {
        for value in values.iter() {
            for i in 0..value.num_elements() {
                let element = value.get_element(i);
                if !element.is_integer() {
                    eprintln!(
                        "Expected array of {}-bit integer attributes for array of integer arrays",
                        width,
                    );
                    exit(ExitCode::IRError);
                }
                let t = IntegerType::from(*element.get_type().get());
                if t.get_width() != width {
                    eprintln!(
                        "Expected array of {}-bit integer attributes for array of integer arrays",
                        width,
                    );
                    exit(ExitCode::IRError);
                }
            }
        }
        let v: Vec<Attribute> = values.iter().map(|a| a.as_attribute()).collect();
        Self::from(*Array::new(context, &v).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_array() {
            eprintln!("Expected array attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_integer_arrays(&self) -> Vec<Array> {
        let args = self.as_array();
        (0..args.num_elements())
            .map(|i| Array::from(*args.get_element(i).get()))
            .collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }

    fn num_elements_flattened(&self) -> isize {
        let v = self.as_integer_arrays();
        v.iter().fold(0, |acc, a| acc + a.num_elements())
    }
}

pub trait NamedBool: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, value: bool) -> Self {
        Self::from(*BoolAttr::new(context, value).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_bool() {
            eprintln!("Expected bool attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_bool(&self) -> BoolAttr {
        BoolAttr::from(*self.get())
    }

    fn get_value(&self) -> bool {
        self.as_bool().get_value()
    }
}

pub trait NamedFloatOrIndexOrInteger: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new_float(attr: &FloatAttr) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn new_index(attr: &IndexAttr) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn new_integer(attr: &IntegerAttr) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_float() && !attr_.is_integer() && !attr_.is_index() {
            eprintln!("Expected float, index, or integer attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_float(&self) -> Option<FloatAttr> {
        if self.is_float() {
            Some(FloatAttr::from(*self.get()))
        } else {
            None
        }
    }

    fn as_index(&self) -> Option<IndexAttr> {
        if self.is_index() {
            Some(IndexAttr::from(*self.get()))
        } else {
            None
        }
    }

    fn as_integer(&self) -> Option<IntegerAttr> {
        if self.is_integer() {
            Some(IntegerAttr::from(*self.get()))
        } else {
            None
        }
    }

    fn is_float(&self) -> bool {
        self.as_attribute().is_float()
    }

    fn is_index(&self) -> bool {
        self.as_attribute().is_index()
    }

    fn is_integer(&self) -> bool {
        self.as_attribute().is_integer()
    }
}

pub trait NamedFunction: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(f: &Function) -> Self {
        let attr = TypeAttr::new(&f.as_type());
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let f = Self::from(attr);
        let a = f.as_attribute();
        if !a.is_type() || !TypeAttr::from(*a.get()).get_type().is_function() {
            eprintln!("Expected typed function attribute");
            exit(ExitCode::DialectError);
        }
        f
    }

    fn as_type_attribute(&self) -> TypeAttr {
        TypeAttr::from(*self.get())
    }

    fn get_function(&self) -> Function {
        Function::from(*self.as_type_attribute().get_type().get())
    }
}

pub trait NamedI32DenseArray: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, values: &[i32]) -> Self {
        Self::from(*DenseArray::new_i32(context, values).get_mut())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_dense_array_i32() {
            eprintln!("Expected dense array of 32-bit integers attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_dense_array(&self) -> DenseArray {
        DenseArray::from((*self.get(), DenseArrayLayout::I32))
    }

    fn num_elements(&self) -> isize {
        self.as_dense_array().num_elements()
    }
}

pub trait NamedI64DenseArray: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, values: &[i64]) -> Self {
        Self::from(*DenseArray::new_i64(context, values).get_mut())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_dense_array_i64() {
            eprintln!("Expected dense array of 64-bit integers attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_dense_array(&self) -> DenseArray {
        DenseArray::from((*self.get(), DenseArrayLayout::I64))
    }

    fn num_elements(&self) -> isize {
        self.as_dense_array().num_elements()
    }
}

pub trait NamedInitialization: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(attr: &Attribute) -> Self {
        Self::from_checked(*attr.get())
    }

    fn new_elements(elements: &Elements) -> Self {
        Self::new(&elements.as_attribute())
    }

    fn new_uninitialized(context: &Context) -> Self {
        Self::new(&Unit::new(context).as_attribute())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_unit() && !attr_.as_attribute().is_elements() {
            eprintln!("Expected unit or elements attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn get_elements(&self) -> Option<Elements> {
        if self.is_initialized() && self.as_attribute().is_elements() {
            Some(Elements::from(*self.get()))
        } else {
            None
        }
    }

    fn is_initialized(&self) -> bool {
        !self.as_attribute().is_unit()
    }
}

pub trait NamedIndex: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, n: i64) -> Self {
        Self::from(*IndexAttr::new(context, n).get_mut())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_index() {
            eprintln!("Expected index attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_index(&self) -> IndexAttr {
        IndexAttr::from(*self.get())
    }
}

pub trait NamedInteger: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, n: i64, width: usize) -> Self {
        let t = IntegerType::new(context, width);
        Self::from(*IntegerAttr::new(&t, n).get())
    }

    fn new_index(context: &Context, n: i64) -> Self {
        Self::from(*IntegerAttr::new_index(context, n).get())
    }

    fn new_signed(context: &Context, n: i64, width: usize) -> Self {
        let t = IntegerType::new_signed(context, width);
        Self::from(*IntegerAttr::new(&t, n).get())
    }

    fn new_signless(context: &Context, n: i64, width: usize) -> Self {
        let t = IntegerType::new_signless(context, width);
        Self::from(*IntegerAttr::new(&t, n).get())
    }

    fn new_unsigned(context: &Context, n: i64, width: usize) -> Self {
        let t = IntegerType::new_unsigned(context, width);
        Self::from(*IntegerAttr::new(&t, n).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_integer() {
            eprintln!("Expected integer attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_integer(&self) -> IntegerAttr {
        IntegerAttr::from(*self.get())
    }

    fn get_value(&self) -> i64 {
        self.as_integer().get_int()
    }
}

pub trait NamedMemoryLayout: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new_affine_map(attr: &AffineMap) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn new_strided_layout(attr: &StridedLayout) -> Self {
        Self::from(*attr.get())
    }

    fn is_affine_map(&self) -> bool {
        self.as_attribute().is_affine_map()
    }

    fn is_strided_layout(&self) -> bool {
        self.as_attribute().is_strided_layout()
    }

    fn as_affine_map(&self) -> Option<AffineMap> {
        if self.is_affine_map() {
            Some(AffineMap::from(*self.get()))
        } else {
            None
        }
    }

    fn as_strided_layout(&self) -> Option<StridedLayout> {
        if self.is_strided_layout() {
            Some(StridedLayout::from(*self.get()))
        } else {
            None
        }
    }
}

pub trait NamedMemorySpace: From<MlirAttribute> + IAttributeNamed + cmp::PartialEq + Sized {
    fn from_checked(attr: MlirAttribute) -> Self;

    fn new_integer(attr: &IntegerAttr) -> Self {
        Self::from(*attr.get())
    }

    fn new_none() -> Self {
        Self::from(*Attribute::new().get())
    }

    fn new_opaque(attr: &Opaque) -> Self {
        Self::from(*attr.get())
    }

    fn new_string(attr: &StringAttr) -> Self {
        Self::from(*attr.get())
    }

    fn as_integer(&self) -> Option<IntegerAttr> {
        match self.is_integer() {
            false => None,
            true => Some(IntegerAttr::from(*self.get())),
        }
    }

    fn as_none(&self) -> Option<Attribute> {
        match self.is_none() {
            false => None,
            true => Some(Attribute::from(*self.get())),
        }
    }

    fn as_opaque(&self) -> Option<Opaque> {
        match self.is_opaque() {
            false => None,
            true => Some(Opaque::from(*self.get())),
        }
    }

    fn as_string(&self) -> Option<StringAttr> {
        match self.is_string() {
            false => None,
            true => Some(StringAttr::from(*self.get())),
        }
    }

    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }

    fn is_integer(&self) -> bool {
        self.as_attribute().is_integer()
    }

    fn is_none(&self) -> bool {
        self.as_attribute().is_null()
    }

    fn is_opaque(&self) -> bool {
        self.as_attribute().is_opaque()
    }

    fn is_string(&self) -> bool {
        self.as_attribute().is_string()
    }
}

pub trait NamedMemRef: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(t: &MemRef) -> Self {
        Self::from(*TypeAttr::new(&t.as_type()).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_type() {
            eprintln!("Expected type attribute");
            exit(ExitCode::IRError);
        }
        let attr_typed = TypeAttr::from(attr);
        if !attr_typed.get_type().is_memref() {
            eprintln!("Expected memory reference type attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_type(&self) -> TypeAttr {
        TypeAttr::from(*self.get())
    }
}

pub trait NamedOpaque: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(t: &Type, namespace: &StringRef, data: &StringRef) -> Self {
        Self::from(*Opaque::new(t, namespace, data).get())
    }

    fn new_custom(t: &Type, cad: &CustomAttributeData) -> Self {
        let namespace = StringBacked::from(cad.get_namespace());
        let data = StringBacked::from(format!("{}{}", cad.get_name(), cad.get_data_joined()));
        Self::new(t, &namespace.as_string_ref(), &data.as_string_ref())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_opaque() {
            eprintln!("Expected opqaue attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_opaque(&self) -> Opaque {
        Opaque::from(*self.get())
    }
}

pub trait NamedParsed: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, s: &StringRef) -> Self {
        Self::from(*Attribute::from_parse(context, s).get())
    }

    fn new_custom(context: &Context, cad: &CustomAttributeData) -> Self {
        let s = StringBacked::from(cad.to_string());
        Self::new(context, &s.as_string_ref())
    }
}

pub trait NamedPermutation: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, permutation: &mut [c_uint]) -> Self {
        let map = AffineMap::new_permutation(context, permutation);
        Self::from(*map.as_attribute().get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_affine_map() {
            eprintln!("Expected affine map attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_affine_map(&self) -> AffineMap {
        AffineMap::from_attribute(&Attribute::from(*self.get()))
    }
}

pub trait NamedString: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, s: &StringRef) -> Self {
        let s_ = StringAttr::new(context, s);
        Self::from(*s_.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_string() {
            eprintln!("Expected string attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_string(&self) -> StringAttr {
        StringAttr::from(*self.get())
    }
}

pub trait NamedSymbolRef: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context, s: &StringRef) -> Self {
        let s_ = SymbolRef::new_flat(context, s);
        Self::from(*s_.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_flat_symbol_ref() {
            eprintln!("Expected flat named symbol reference");
            exit(ExitCode::DialectError);
        }
        attr_
    }

    fn from_symbol_ref(s_ref: &SymbolRef) -> Self {
        Self::from_checked(*s_ref.get())
    }

    fn as_string_ref(&self) -> StringRef {
        match self.as_symbol_ref().get_value() {
            Some(s) => s,
            None => {
                eprintln!("Expected flat named symbol reference");
                exit(ExitCode::DialectError);
            }
        }
    }

    fn as_symbol_ref(&self) -> SymbolRef {
        SymbolRef::from(*self.get())
    }
}

pub trait NamedType: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(t: &Type) -> Self {
        Self::from(*TypeAttr::new(t).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_type() {
            eprintln!("Expected type attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_type(&self) -> TypeAttr {
        TypeAttr::from(*self.get())
    }
}

pub trait NamedUnit: From<MlirAttribute> + IAttributeNamed + Sized {
    fn new(context: &Context) -> Self {
        Self::from(*Unit::new(context).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.as_attribute().is_unit() {
            eprintln!("Expected unit attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_unit(&self) -> Unit {
        Unit::from(*self.get())
    }
}

///////////////////////////////
//  Support Implementation
///////////////////////////////

impl CustomAttributeData {
    pub fn new(name: String, namespace: String, data: Vec<String>) -> Self {
        Self {
            name,
            namespace,
            data,
        }
    }

    pub fn get_data(&self) -> &[String] {
        &self.data
    }

    pub fn get_data_joined(&self) -> String {
        if self.data.is_empty() {
            String::default()
        } else {
            format!("<{}>", self.data.join(","))
        }
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_namespace(&self) -> &String {
        &self.namespace
    }

    fn is_angle_bracket(c: char) -> bool {
        matches!(c, '<' | '>')
    }

    fn is_prefix(s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        !chars.is_empty() && chars[0] == '#' && !chars.iter().any(|&c| Self::is_angle_bracket(c))
    }

    fn split_prefix(s: &str) -> Result<(String, String), &'static str> {
        let s_ = &s[1..];
        let parts: Vec<&str> = s_.split('.').collect();
        if parts.len() == 2 {
            Ok((parts[0].to_string(), parts[1].to_string()))
        } else {
            Err("Expected prefix of form '#namespace.name'")
        }
    }
}

impl fmt::Display for CustomAttributeData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "#{}.{}{}",
            self.get_namespace(),
            self.get_name(),
            self.get_data_joined()
        )
    }
}

impl From<String> for CustomAttributeData {
    fn from(s: String) -> Self {
        Self::from(&s)
    }
}

impl From<&String> for CustomAttributeData {
    fn from(s: &String) -> Self {
        match Self::from_str(s.as_str()) {
            Ok(cad) => cad,
            Err(msg) => {
                eprintln!("{}", msg);
                exit(ExitCode::IRError);
            }
        }
    }
}

impl From<StringRef> for CustomAttributeData {
    fn from(s_ref: StringRef) -> Self {
        Self::from(&s_ref.to_string())
    }
}

impl From<&StringRef> for CustomAttributeData {
    fn from(s_ref: &StringRef) -> Self {
        Self::from(s_ref.to_string())
    }
}

impl FromStr for CustomAttributeData {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(Self::is_angle_bracket).collect();
        if parts.is_empty() || !Self::is_prefix(parts[0]) {
            Err("Expected prefix (e.g., #namespace.name) for custom attribute data")?
        }
        let (namespace, name) = Self::split_prefix(parts[0])?;
        Ok(Self::new(
            namespace,
            name,
            parts[0..1].iter().map(|s_| s_.to_string()).collect(),
        ))
    }
}
