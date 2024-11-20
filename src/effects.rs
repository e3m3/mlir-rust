// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

extern crate mlir_sys as mlir;

use std::fmt;

pub type MemoryEffectList = &'static [&'static MemoryEffect];

#[derive(Clone,Copy,Default,PartialEq)]
pub enum MemoryEffectType {
    #[default]
    None,
    Write,
}

#[derive(Clone,Copy,Default,PartialEq)]
pub enum SideEffectResource {
    #[default]
    DefaultResource,
}

#[derive(Clone,Copy,Default,PartialEq)]
pub struct MemoryEffect(MemoryEffectType, SideEffectResource);

pub const MEFF_DEFAULT_WRITE: &MemoryEffect = &MemoryEffect::default_write();
pub const MEFF_NO_MEMORY_EFFECT: &MemoryEffect = &MemoryEffect::no_memory_effect();

impl MemoryEffect {
    #[inline]
    pub const fn default_write() -> Self {
        MemoryEffect(MemoryEffectType::Write, SideEffectResource::DefaultResource)
    }

    #[inline]
    pub const fn no_memory_effect() -> Self {
        MemoryEffect(MemoryEffectType::None, SideEffectResource::DefaultResource)
    }

    pub fn from(effect: MemoryEffectType, resource: SideEffectResource) -> Self {
        MemoryEffect(effect, resource)
    }

    pub fn get_resource(&self) -> SideEffectResource {
        self.1
    }

    pub fn get_type(&self) -> MemoryEffectType {
        self.0
    }
}

impl fmt::Display for MemoryEffectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            MemoryEffectType::None      => "None",
            MemoryEffectType::Write     => "Write",
        })
    }
}

impl fmt::Display for SideEffectResource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            SideEffectResource::DefaultResource => "DefaultResource",
        })
    }
}

impl fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MemoryEffects::Effect{{MemoryEffects::{} on ::mlir::SideEffects::{}}}",
            self.get_type(),
            self.get_resource(),
        )
    }
}
