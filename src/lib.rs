// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

pub mod attributes;
pub mod dialects;
pub mod exit_code;
pub mod ir;
pub mod types;

#[macro_export]
macro_rules! do_unsafe {
    ($expression:expr) => {
        unsafe { $expression }
    }
}
