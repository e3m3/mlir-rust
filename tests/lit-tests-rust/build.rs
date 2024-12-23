// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#[path = "src/default_test.rs"]
mod default_test;
use crate::default_test::DEFAULT_TEST_NAME;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process;

const TEST_EXT: &str = "lit-rs";
const TESTS_CRATE_BODY: &str = r#"
// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause
//
// THIS FILE IS AUTO-GENERATED (BY `build.rs`). DO NOT EDIT!

mod test_not_found {
    use std::env;
    use crate::common::TestResult;

    pub fn test() -> TestResult {
        const MSG_MISSING: &str = "MISSING ARG (<test-name>)";
        let args: Vec<String> = env::args().collect();
        let test_name = args.last().cloned().unwrap_or(MSG_MISSING.to_string());
        Err(format!("Test '{}' not found", test_name))
    }
}
"#;

const TESTS_REGISTRY_HEADER: &str = r#"
fn register_test_callbacks(test_registry: &mut TestRegistry) -> () {
    test_registry.add_test(DEFAULT_TEST_NAME, crate::test_not_found::test);
"#;

enum ExitCode {
    Ok = 0,
    BuildError = 100,
}

fn append_module(s: &mut String, path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let test_name = test_name(path);
    let path_decl = format!("#[path = \"{}\"]", path.display());
    let mod_decl = format!("mod {};", test_name);
    s.push('\n');
    s.push_str(&path_decl);
    s.push('\n');
    s.push_str(&mod_decl);
    s.push('\n');
    true
}

fn append_register_test_callback(s: &mut String, path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let name = test_name(path);
    let call_site = format!(
        "    test_registry.add_test(\"{}\", crate::{}::test);",
        name, name
    );
    s.push_str(&call_site);
    s.push('\n');
    true
}

fn exit(code: ExitCode) -> ! {
    process::exit(code as i32);
}

fn manifest_path() -> PathBuf {
    let root_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(root_dir.as_str())
}

#[allow(dead_code)]
fn input(file: &str) -> PathBuf {
    input_path().join(file)
}

fn input_path() -> PathBuf {
    manifest_path().join("src")
}

fn output(file: &str) -> PathBuf {
    output_path().join(file)
}

fn output_path() -> PathBuf {
    let build_profile = env::var("PROFILE").unwrap();
    let mut root_path = manifest_path();
    assert!(root_path.pop());
    assert!(root_path.pop());
    root_path.join("target").join(build_profile)
}

fn path_is_test(path: &Path) -> bool {
    path.is_file() && path.extension().unwrap_or_default() == TEST_EXT
}

fn test_name(path: &Path) -> String {
    assert!(path.is_file());
    path.file_stem()
        .unwrap_or(&OsString::from(DEFAULT_TEST_NAME.to_string()))
        .to_str()
        .unwrap_or(DEFAULT_TEST_NAME)
        .to_string()
}

fn write_file_and_exit(path: &str, body: &str) -> ! {
    match fs::write(path, body) {
        Ok(()) => exit(ExitCode::Ok),
        Err(msg) => {
            eprintln!("{}", msg);
            exit(ExitCode::BuildError);
        }
    }
}

fn main() -> ! {
    println!("cargo::rerun-if-changed=src");

    let mut tests_crate_body = TESTS_CRATE_BODY[1..].to_string();
    let mut tests_registry_body = TESTS_REGISTRY_HEADER[1..].to_string();

    let path_in = input_path();
    let path_out_ = output("lit.rs");
    let path_out = path_out_.as_os_str().to_str().unwrap_or_default();
    let Ok(entries) = fs::read_dir(&path_in) else {
        eprintln!(
            "Failed to read files from input path '{}'",
            path_in.display()
        );
        exit(ExitCode::BuildError);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path_is_test(&path) {
            continue; // Skip non-test files (e.g., `main.rs`)
        } else if !append_module(&mut tests_crate_body, &path) {
            eprintln!("Failed to add test module '{}'", path.display());
            exit(ExitCode::BuildError);
        }
        append_register_test_callback(&mut tests_registry_body, &path);
    }
    tests_crate_body.push('\n');
    tests_crate_body.push_str(&tests_registry_body);
    tests_crate_body.push('}');
    write_file_and_exit(path_out, &tests_crate_body);
}
