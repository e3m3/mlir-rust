// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use std::ffi::c_int;
use std::ffi::c_uint;

use crate::attributes;
use crate::dialects;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::array::Array;
use attributes::bool::Bool as BoolAttr;
use attributes::dense_array::DenseArray;
use attributes::dense_array::Layout as DenseArrayLayout;
use attributes::dictionary::Dictionary;
use attributes::elements::Elements;
use attributes::float::Float as FloatAttr;
use attributes::IRAttribute;
use attributes::IRAttributeNamed;
use attributes::integer::Integer as IntegerAttr;
use attributes::string::String as StringAttr;
use attributes::symbol_ref::SymbolRef;
use attributes::r#type::Type as TypeAttr;
use attributes::unit::Unit;
use dialects::affine::Map as AffineMap;
use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::StringRef;
use ir::Type;
use types::function::Function;
use types::integer::Integer as IntegerType;
use types::IRType;
use types::mem_ref::MemRef;

///////////////////////////////
//  Specialized Traits
///////////////////////////////

pub trait NamedArrayOfBools: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, elements: &[BoolAttr]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes for array of bools");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_bool()) {
            eprintln!("Expected array of dictionary attributes for array of bools");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_bools(&self) -> Vec<BoolAttr> {
        let args = self.as_array();
        (0..args.num_elements()).map(|i| BoolAttr::from(*args.get_element(i).get())).collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfDictionaries: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, elements: &[Dictionary]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes for array of dictionaries");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_dictionary()) {
            eprintln!("Expected array of dictionary attributes for array of dictionaries");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_dictionaries(&self) -> Vec<Dictionary> {
        let args = self.as_array();
        (0..args.num_elements()).map(|i| Dictionary::from(*args.get_element(i).get())).collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfIntegers: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, elements: &[IntegerAttr]) -> Self {
        let e: Vec<Attribute> = elements.iter().map(|e| e.as_attribute()).collect();
        let attr = Array::new(context, &e);
        Self::from(*attr.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let args = Self::from(attr);
        if !args.as_attribute().is_array() {
            eprintln!("Expected array of dictionary attributes for array of integers");
            exit(ExitCode::IRError);
        }
        let args_array = args.as_array();
        if (0..args_array.num_elements()).any(|i| args_array.get_element(i).is_integer()) {
            eprintln!("Expected array of dictionary attributes for array of integers");
            exit(ExitCode::IRError);
        }
        args
    }

    fn as_array(&self) -> Array {
        Array::from(*self.get())
    }

    fn as_integers(&self) -> Vec<IntegerAttr> {
        let args = self.as_array();
        (0..args.num_elements()).map(|i| IntegerAttr::from(*args.get_element(i).get())).collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }
}

pub trait NamedArrayOfIntegerArrays: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, values: &[Array], width: c_uint) -> Self {
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
        (0..args.num_elements()).map(|i| Array::from(*args.get_element(i).get())).collect()
    }

    fn num_elements(&self) -> isize {
        self.as_array().num_elements()
    }

    fn num_elements_flattened(&self) -> isize {
        let v = self.as_integer_arrays();
        v.iter().fold(0, |acc,a| acc + a.num_elements())
    }
}

pub trait NamedBool: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, value: c_int) -> Self {
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

pub trait NamedFloatOrInteger: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new_float(attr: &FloatAttr) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn new_integer(attr: &IntegerAttr) -> Self {
        Self::from(*attr.as_attribute().get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        if !attr_.is_float() && !attr_.is_integer() {
            eprintln!("Expected float or integer attribute");
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

    fn is_integer(&self) -> bool {
        self.as_attribute().is_integer()
    }
}

pub trait NamedFunction: From<MlirAttribute> + IRAttributeNamed + Sized {
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

pub trait NamedI64DenseArray: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, values: &[i64]) -> Self {
        Self::from(*DenseArray::new_i64(context, values).get())
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
        DenseArray::from(*self.get(), DenseArrayLayout::I64)
    }

    fn num_elements(&self) -> isize {
        self.as_dense_array().num_elements()
    }
}

pub trait NamedInitialization: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(attr: &Attribute) -> Self {
        Self::from(*attr.get())
    }

    fn new_elements(elements: &Elements) -> Self {
        Self::new(&elements.as_attribute())
    }

    fn new_uninitialized(context: &Context) -> Self {
        Self::new(&Unit::new(context).as_attribute())
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

pub trait NamedInteger: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, n: i64, width: c_uint) -> Self {
        let t = IntegerType::new_signless(context, width);
        Self::from(*IntegerAttr::new(&t.as_type(), n).get())
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

pub trait NamedMemRef: From<MlirAttribute> + IRAttributeNamed + Sized {
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
        if !attr_typed.get_type().is_mem_ref() {
            eprintln!("Expected memory reference type attribute");
            exit(ExitCode::IRError);
        }
        attr_
    }

    fn as_type(&self) -> TypeAttr {
        TypeAttr::from(*self.get())
    }
}

pub trait NamedPermutation: From<MlirAttribute> + IRAttributeNamed + Sized {
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

pub trait NamedString: From<MlirAttribute> + IRAttributeNamed + Sized {
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

pub trait NamedSymbolRef: From<MlirAttribute> + IRAttributeNamed + Sized {
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

    fn as_string_ref(&self) -> StringRef {
        match self.as_symbol_ref().get_value() {
            Some(s) => s,
            None    => {
                eprintln!("Expected flat named symbol reference");
                exit(ExitCode::DialectError);
            },
        }
    }

    fn as_symbol_ref(&self) -> SymbolRef {
        SymbolRef::from(*self.get())
    }
}

pub trait NamedType: From<MlirAttribute> + IRAttributeNamed + Sized {
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

pub trait NamedUnit: From<MlirAttribute> + IRAttributeNamed + Sized {
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
