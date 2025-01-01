// Copyright 2024-2025, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use mlir_sys::MlirType;
use mlir_sys::mlirUnrankedMemRefTypeGet;
use mlir_sys::mlirUnrankedMemRefTypeGetChecked;
use mlir_sys::mlirUnrankedMemRefTypeGetTypeID;
use mlir_sys::mlirUnrankedMemrefGetMemorySpace;

use crate::attributes;
use crate::do_unsafe;
use crate::exit_code;
use crate::ir;
use crate::types;

use attributes::specialized::NamedMemorySpace;
use exit_code::ExitCode;
use exit_code::exit;
use ir::Location;
use ir::Type;
use ir::TypeID;
use types::IType;
use types::shaped::Shaped;

#[derive(Clone)]
pub struct UnrankedMemRef(MlirType);

impl UnrankedMemRef {
    pub fn new(t: &Type, memory_space: &impl NamedMemorySpace) -> Self {
        Self::from(do_unsafe!(mlirUnrankedMemRefTypeGet(
            *t.get(),
            *memory_space.get()
        )))
    }

    pub fn new_checked(t: &Type, memory_space: &impl NamedMemorySpace, loc: &Location) -> Self {
        Self::from(do_unsafe!(mlirUnrankedMemRefTypeGetChecked(
            *loc.get(),
            *t.get(),
            *memory_space.get()
        )))
    }

    pub fn as_shaped(&self) -> Shaped {
        Shaped::from(*self.get())
    }

    pub fn from_type(t: &Type) -> Self {
        if !t.is_unranked_memref() {
            eprintln!("Cannot coerce type to unranked mem ref type: {}", t);
            exit(ExitCode::IRError);
        }
        Self(*t.get())
    }

    pub fn get(&self) -> &MlirType {
        &self.0
    }

    pub fn get_memory_space<T: NamedMemorySpace>(&self) -> T {
        // Why is this capitalized differently?
        T::from_checked(do_unsafe!(mlirUnrankedMemrefGetMemorySpace(*self.get())))
    }

    pub fn get_mut(&mut self) -> &mut MlirType {
        &mut self.0
    }

    pub fn get_type_id() -> TypeID {
        TypeID::from(do_unsafe!(mlirUnrankedMemRefTypeGetTypeID()))
    }
}

impl From<MlirType> for UnrankedMemRef {
    fn from(t: MlirType) -> Self {
        Self::from(Type::from(t))
    }
}

impl From<Type> for UnrankedMemRef {
    fn from(t: Type) -> Self {
        Self::from(&t)
    }
}

impl From<&Type> for UnrankedMemRef {
    fn from(t: &Type) -> Self {
        Self::from_type(t)
    }
}

impl IType for UnrankedMemRef {
    fn get(&self) -> &MlirType {
        self.get()
    }

    fn get_mut(&mut self) -> &mut MlirType {
        self.get_mut()
    }
}
