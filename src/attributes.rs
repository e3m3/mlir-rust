// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::MlirAttribute;

use std::cmp;
use std::ffi::c_uint;
use std::hint::black_box;

use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Attribute;
use ir::Context;
use ir::Identifier;
use ir::StringBacked;
use ir::StringRef;
use types::function::Function;
use types::integer::Integer as IntegerType;
use types::IRType;

use array::Array;
use dense_array::DenseArray;
use dense_array::Layout as DenseArrayLayout;
use dictionary::Dictionary;
use float::Float as FloatAttr;
use named::Named;
use integer::Integer as IntegerAttr;
use string::String as StringAttr;
use symbol_ref::SymbolRef;
use r#type::Type as TypeAttr;
use unit::Unit;

pub mod array;
pub mod bool;
pub mod dense_array;
pub mod dense_elements;
pub mod dense_resource_elements;
pub mod dictionary;
pub mod elements;
pub mod float;
pub mod integer;
pub mod integer_set;
pub mod named;
pub mod opaque;
pub mod sparse_elements;
pub mod strided_layout;
pub mod string;
pub mod symbol_ref;
pub mod r#type;
pub mod unit;

///////////////////////////////
//  Generic Traits
///////////////////////////////

pub trait IRAttribute {
    fn get(&self) -> &MlirAttribute;
    fn get_mut(&mut self) -> &mut MlirAttribute;

    fn as_attribute(&self) -> Attribute {
        Attribute::from(*self.get())
    }

    fn get_context(&self) -> Context {
        self.as_attribute().get_context()
    }
}

pub trait IRAttributeNamed: IRAttribute {
    fn get_name() -> &'static str where Self: Sized;

    fn as_named_attribute(&self) -> Named where Self: Sized {
        let attr = self.as_attribute();
        let context = attr.get_context();
        let name = StringBacked::from_string(&Self::get_name().to_string());
        let id = Identifier::new(&context, &name.as_string_ref());
        Named::new(&id, &attr.as_attribute())
    }
}

impl cmp::PartialEq for dyn IRAttribute {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

impl cmp::PartialEq for dyn IRAttributeNamed {
    fn eq(&self, rhs: &Self) -> bool {
        self.as_attribute() == rhs.as_attribute()
    }
}

///////////////////////////////
//  Specialized Traits
///////////////////////////////

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

    fn get_type(&self) -> Function {
        Function::from(*self.as_type_attribute().get_type().get())
    }
}

pub trait NamedI64DenseArray: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, values: &[i64]) -> Self {
        Self::from(*DenseArray::new_i64(context, values).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        let _ = black_box(attr_.as_dense_array()); // Don't use the dense array, but check it.
        attr_
    }

    fn as_dense_array(&self) -> DenseArray {
        DenseArray::from(*self.get(), DenseArrayLayout::I64)
    }
}

pub trait NamedInteger: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, n: i64, width: c_uint) -> Self {
        let t = IntegerType::new_signless(context, width);
        Self::from(*IntegerAttr::new(&t.as_type(), n).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        let _ = black_box(attr_.as_integer()); // Don't use the integer, but check it.
        attr_
    }

    fn as_integer(&self) -> IntegerAttr {
        IntegerAttr::from(*self.get())
    }

    fn get_value(&self) -> i64 {
        self.as_integer().get_int()
    }
}

pub trait NamedString: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context, s: &StringRef) -> Self {
        let s_ = StringAttr::new(context, s);
        Self::from(*s_.get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        let _ = black_box(attr_.as_string()); // Don't use the string, but check it.
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

pub trait NamedUnit: From<MlirAttribute> + IRAttributeNamed + Sized {
    fn new(context: &Context) -> Self {
        Self::from(*Unit::new(context).get())
    }

    fn from_checked(attr: MlirAttribute) -> Self {
        let attr_ = Self::from(attr);
        let _ = black_box(attr_.as_unit()); // Don't use the unit, but check it.
        attr_
    }

    fn as_unit(&self) -> Unit {
        Unit::from(*self.get())
    }
}
