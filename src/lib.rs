// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

pub mod attributes;
pub mod dialects;
pub mod effects;
pub mod exit_code;
pub mod interfaces;
pub mod ir;
pub mod traits;
pub mod types;

#[macro_export]
macro_rules! do_unsafe {
    ($expression:expr) => {
        unsafe { $expression }
    };
}
