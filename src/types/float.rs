// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use mlir::mlirFloatTypeGetWidth;
use mlir::mlirFloat8E5M2TypeGet;
use mlir::mlirFloat8E5M2TypeGetTypeID;
use mlir::mlirFloat8E4M3TypeGet;
use mlir::mlirFloat8E4M3TypeGetTypeID;
use mlir::mlirFloat8E4M3FNTypeGet;
use mlir::mlirFloat8E4M3FNTypeGetTypeID;
use mlir::mlirFloat8E5M2FNUZTypeGet;
use mlir::mlirFloat8E5M2FNUZTypeGetTypeID;
use mlir::mlirFloat8E4M3FNUZTypeGet;
use mlir::mlirFloat8E4M3FNUZTypeGetTypeID;
use mlir::mlirFloat8E4M3B11FNUZTypeGet;
use mlir::mlirFloat8E4M3B11FNUZTypeGetTypeID;
use mlir::mlirBF16TypeGet;
use mlir::mlirBFloat16TypeGetTypeID;
use mlir::mlirF16TypeGet;
use mlir::mlirFloat16TypeGetTypeID;
use mlir::mlirF32TypeGet;
use mlir::mlirFloat32TypeGetTypeID;
use mlir::mlirF64TypeGet;
use mlir::mlirFloat64TypeGetTypeID;
use mlir::mlirTF32TypeGet;
use mlir::mlirFloatTF32TypeGetTypeID;
use mlir::mlirTypeIsAFloat8E5M2;
use mlir::mlirTypeIsAFloat8E4M3;
use mlir::mlirTypeIsAFloat8E4M3FN;
use mlir::mlirTypeIsAFloat8E5M2FNUZ;
use mlir::mlirTypeIsAFloat8E4M3FNUZ;
use mlir::mlirTypeIsAFloat8E4M3B11FNUZ;
use mlir::mlirTypeIsABF16;
use mlir::mlirTypeIsAF16;
use mlir::mlirTypeIsAF32;
use mlir::mlirTypeIsAF64;
use mlir::mlirTypeIsATF32;
use mlir::MlirType;

use std::ffi::c_uint;
use std::fmt;

use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use exit_code::exit;
use exit_code::ExitCode;
use ir::Context;
use ir::Type;
use ir::TypeID;
use types::IRType;

#[derive(Clone)]
pub struct Float(MlirType, Layout);

#[derive(Clone,Copy,PartialEq)]
pub enum Layout {
    F8E5M2,
    F8E4M3,
    F8E4M3FN,
    F8E5M2FNUZ,
    F8E4M3FNUZ,
    F8E4M3B11FNUZ,
    BF16,
    F16,
    F32,
    F64,
    TF32,
}

impl Float {
    pub fn new(context: &Context, layout: Layout) -> Self {
        let t = do_unsafe!(match layout {
            Layout::F8E5M2          => mlirFloat8E5M2TypeGet(*context.get()),
            Layout::F8E4M3          => mlirFloat8E4M3TypeGet(*context.get()),
            Layout::F8E4M3FN        => mlirFloat8E4M3FNTypeGet(*context.get()),
            Layout::F8E5M2FNUZ      => mlirFloat8E5M2FNUZTypeGet(*context.get()),
            Layout::F8E4M3FNUZ      => mlirFloat8E4M3FNUZTypeGet(*context.get()),
            Layout::F8E4M3B11FNUZ   => mlirFloat8E4M3B11FNUZTypeGet(*context.get()),
            Layout::BF16            => mlirBF16TypeGet(*context.get()),
            Layout::F16             => mlirF16TypeGet(*context.get()),
            Layout::F32             => mlirF32TypeGet(*context.get()),
            Layout::F64             => mlirF64TypeGet(*context.get()),
            Layout::TF32            => mlirTF32TypeGet(*context.get()),
        });
        Self::from(t, layout)
    }

    pub fn from(t: MlirType, layout: Layout) -> Self {
        let f = Float(t, layout);
        if !f.is(layout) {
            eprint!("Cannot coerce type to float type with layout '{}': ", layout);
            f.as_type().dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        f
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_float() {
            eprint!("Cannot coerce type to float type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        }
        let mut f = Self::from(*t.get(), Layout::F32); // Unused layout
        f.1 = if f.is_f8_e5_m2() {
            Layout::F8E5M2
        } else if f.is_f8_e4_m3() {
            Layout::F8E4M3
        } else if f.is_f8_e4_m3_fn() {
            Layout::F8E4M3FN
        } else if f.is_f8_e5_m2_fnuz() {
            Layout::F8E5M2FNUZ
        } else if f.is_f8_e4_m3_fnuz() {
            Layout::F8E4M3FNUZ
        } else if f.is_f8_e4_m3_b11_fnuz() {
            Layout::F8E4M3B11FNUZ
        } else if f.is_bf16() {
            Layout::BF16
        } else if f.is_f16() {
            Layout::F16
        } else if f.is_f32() {
            Layout::F32
        } else if f.is_f64() {
            Layout::F64
        } else if f.is_tf32() {
            Layout::TF32
        } else {
            eprint!("Unexpected float layout for type: ");
            t.dump();
            eprintln!();
            exit(ExitCode::IRError);
        };
        f
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_layout(&self) -> Layout {
        self.1
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id(layout: Layout) -> TypeID {
        TypeID::from(do_unsafe!(match layout {
            Layout::F8E5M2          => mlirFloat8E5M2TypeGetTypeID(),
            Layout::F8E4M3          => mlirFloat8E4M3TypeGetTypeID(),
            Layout::F8E4M3FN        => mlirFloat8E4M3FNTypeGetTypeID(),
            Layout::F8E5M2FNUZ      => mlirFloat8E5M2FNUZTypeGetTypeID(),
            Layout::F8E4M3FNUZ      => mlirFloat8E4M3FNUZTypeGetTypeID(),
            Layout::F8E4M3B11FNUZ   => mlirFloat8E4M3B11FNUZTypeGetTypeID(),
            Layout::BF16            => mlirBFloat16TypeGetTypeID(),
            Layout::F16             => mlirFloat16TypeGetTypeID(),
            Layout::F32             => mlirFloat32TypeGetTypeID(),
            Layout::F64             => mlirFloat64TypeGetTypeID(),
            Layout::TF32            => mlirFloatTF32TypeGetTypeID(),
        }))
    }

    pub fn get_width(&self) -> c_uint {
        do_unsafe!(mlirFloatTypeGetWidth(self.0))
    }

    pub fn is(&self, layout: Layout) -> bool {
        match layout {
            Layout::F8E5M2          => self.is_f8_e5_m2(),
            Layout::F8E4M3          => self.is_f8_e4_m3(),
            Layout::F8E4M3FN        => self.is_f8_e4_m3_fn(),
            Layout::F8E5M2FNUZ      => self.is_f8_e5_m2_fnuz(),
            Layout::F8E4M3FNUZ      => self.is_f8_e4_m3_fnuz(),
            Layout::F8E4M3B11FNUZ   => self.is_f8_e4_m3_b11_fnuz(),
            Layout::BF16            => self.is_bf16(),
            Layout::F16             => self.is_f16(),
            Layout::F32             => self.is_f32(),
            Layout::F64             => self.is_f64(),
            Layout::TF32            => self.is_tf32(),
        }
    }

    pub fn is_f8_e5_m2(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E5M2(self.0))
    }

    pub fn is_f8_e4_m3(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E4M3(self.0))
    }

    pub fn is_f8_e4_m3_fn(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E4M3FN(self.0))
    }

    pub fn is_f8_e5_m2_fnuz(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E5M2FNUZ(self.0))
    }

    pub fn is_f8_e4_m3_fnuz(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E4M3FNUZ(self.0))
    }

    pub fn is_f8_e4_m3_b11_fnuz(&self) -> bool {
        do_unsafe!(mlirTypeIsAFloat8E4M3B11FNUZ(self.0))
    }

    pub fn is_bf16(&self) -> bool {
        do_unsafe!(mlirTypeIsABF16(self.0))
    }

    pub fn is_f16(&self) -> bool {
        do_unsafe!(mlirTypeIsAF16(self.0))
    }

    pub fn is_f32(&self) -> bool {
        do_unsafe!(mlirTypeIsAF32(self.0))
    }

    pub fn is_f64(&self) -> bool {
        do_unsafe!(mlirTypeIsAF64(self.0))
    }

    pub fn is_tf32(&self) -> bool {
        do_unsafe!(mlirTypeIsATF32(self.0))
    }
}

impl IRType for Float {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Layout::F8E5M2          => "f8E5M2",
            Layout::F8E4M3          => "f8E4M3",
            Layout::F8E4M3FN        => "f8E4M3FN",
            Layout::F8E5M2FNUZ      => "f8E5M2FNUZ",
            Layout::F8E4M3FNUZ      => "f8E4M3FNUZ",
            Layout::F8E4M3B11FNUZ   => "f8E4M3B11FNUZ",
            Layout::BF16            => "bf16",
            Layout::F16             => "f16",
            Layout::F32             => "f32",
            Layout::F64             => "f64",
            Layout::TF32            => "tf32",
        };
        write!(f, "Layout_{}", s)
    }
}
