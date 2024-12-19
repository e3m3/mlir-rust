// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#[cfg(test)]
mod tests {
    extern crate num_cpus;

    use std::env;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process::Command;

    fn path_to_string(path: &Path) -> String {
        String::from(path.to_str().unwrap())
    }

    fn pathbuf_to_string(path: &PathBuf) -> String {
        path.to_str().unwrap().to_string()
    }

    fn get_bin_dir() -> PathBuf {
        env::current_exe().ok().map(|mut path: PathBuf| {
            path.pop();
            path.pop();
            path
        }).unwrap()
    }

    fn get_num_jobs() -> usize {
        num_cpus::get()
    }

    fn get_num_jobs_cargo() -> usize {
        let n = get_num_jobs();
        match env::var("CARGO_JOBS") {
            Ok(j)   => j.parse::<usize>().unwrap_or(n),
            Err(_)  => n,
        }
    }

    fn get_num_jobs_lit() -> usize {
        let n = get_num_jobs();
        match env::var("LIT_JOBS") {
            Ok(j)   => j.parse::<usize>().unwrap_or(n),
            Err(_)  => n,
        }
    }

    fn get_tests_dir() -> PathBuf {
        env::current_exe().ok().map(|mut path: PathBuf| {
            path.pop();
            path.pop();
            path.pop();
            path.pop();
            path.push("tests");
            path
        }).unwrap()
    }

    fn get_shell(os_name: &String) -> String {
        String::from(match os_name.as_str() {
            "linux"     => "bash",
            "macos"     => "bash",
            "windows"   => "cmd",
            _           => {
                eprintln!("Unexpected target_os");
                assert!(false);
                ""
            },
        })
    }

    fn get_lit(os_name: &String) -> String {
        let append_lit: fn(&Path) -> String = |path| {
            String::from(path.join("bin").join("lit").to_str().unwrap())
        };
        match os_name.as_str() {
            "linux"     =>
                match env::var("PYTHON_VENV_PATH") {
                    Ok(path)    => append_lit(Path::new(&path)),
                    Err(_)      => append_lit(Path::new("/usr")),
                },
            "macos"     =>
                match env::var("HOMEBREW_HOME") {
                    Ok(path)    => append_lit(Path::new(&path)),
                    Err(_)      => match env::var("PYTHON_VENV_PATH") {
                        Ok(path)    => append_lit(Path::new(&path)),
                        Err(_)      => append_lit(Path::new("/usr")),
                    }
                },
            "windows"   => {
                match env::var("PYTHON_VENV_PATH") {
                    Ok(path)    => append_lit(Path::new(&path)),
                    Err(_)      => {
                        eprintln!("No supported location for 'lit' found");
                        assert!(false);
                        String::new()
                    },
                }
            },
            _           => {
                eprintln!("OS not supported");
                assert!(false);
                String::new()
            },
        }
    }

    fn get_os() -> String {
        String::from(
            if cfg!(target_os = "linux") {
                "linux"
            } else if cfg!(target_os = "macos") {
                "macos"
            } else if cfg!(target_os = "windows") {
                "windows"
            } else {
                ""
            }
        )
    }

    fn get_arch() -> String {
        String::from(
            if cfg!(target_arch = "x86") {
                "x86"
            } else if cfg!(target_arch = "x86_64") {
                "x86_64"
            } else if cfg!(target_arch = "aarch64") {
                "aarch64"
            } else {
                ""
            }
        )
    }

    #[test]
    fn lit() {
        let os_name: String = get_os();
        let arch: String = get_arch();
        let num_jobs: usize = get_num_jobs();

        #[cfg(debug_assertions)]
        let build_mode = "";
        #[cfg(not(debug_assertions))]
        let build_mode = "--release";

        if os_name.is_empty() {
            eprintln!("Target OS '{}' not yet supported.", os_name);
            assert!(false);
        }
        if arch.is_empty() {
            eprintln!("Target arch '{}' not yet supported.", arch);
            assert!(false);
        }
        eprintln!("Detected {} CPUs/threads.", num_jobs);

        let bin_dir: PathBuf = get_bin_dir();
        let lit_bin_str: String = get_lit(&os_name);
        let lit_bin: &Path = Path::new(lit_bin_str.as_str());
        let shell: String = get_shell(&os_name);
        let tests_dir: PathBuf = get_tests_dir();

        let lit_dir_mlir: PathBuf = tests_dir.join("lit-tests-mlir");
        let lit_dir_rust: PathBuf = tests_dir.join("lit-tests-rust");
        let cfg_path_mlir: PathBuf = lit_dir_mlir.join("lit.cfg");
        let cfg_path_rust: PathBuf = lit_dir_rust.join("lit.cfg");
        let lit_dir_rust_tests_dir: PathBuf = lit_dir_rust.join("src");
        let lit_dir_rust_manifest: PathBuf = lit_dir_rust.join("Cargo.toml");

        println!("Lit binary path: {}", lit_bin_str);

        assert!(bin_dir.is_dir());
        assert!(lit_bin.is_file());
        assert!(lit_dir_mlir.is_dir());
        assert!(lit_dir_rust.is_dir());
        assert!(cfg_path_mlir.is_file());
        assert!(cfg_path_rust.is_file());
        assert!(lit_dir_rust_tests_dir.is_dir());
        assert!(lit_dir_rust_manifest.is_file());

        let bin_dir_str: String = pathbuf_to_string(&bin_dir);
        let lit_dir_mlir_str: String = pathbuf_to_string(&lit_dir_mlir);
        let lit_bin_str: String = path_to_string(&lit_bin);
        let lit_dir_rust_tests_dir_str: String = pathbuf_to_string(&lit_dir_rust_tests_dir);
        let lit_dir_rust_manifest_str: String = pathbuf_to_string(&lit_dir_rust_manifest);

        let env_path_str: String = [
            bin_dir_str.clone(),
        ].join(":");

        let lit_args: String = [
            "--config-prefix=lit",
            "--order=lexical",
            "--show-all",
            format!("--workers={}", get_num_jobs_lit()).as_str(),
            format!("--param=ARCH={}", arch).as_str(),
            format!("--param=OS_NAME={}", os_name).as_str(),
            format!("--param=CARGO_OUTDIR={}", bin_dir_str).as_str(),
            format!("--param=MLIR_TESTDIR={}", lit_dir_mlir_str).as_str(),
            format!("--path={}", env_path_str).as_str(),
        ].join(" ");

        println!("Building lit test binaries from tests in '{}':", lit_dir_rust_tests_dir_str);
        let output_cargo = Command::new(&shell)
            .arg("-c")
            .arg(format!(
                "cargo build {} --manifest-path {} -j {}",
                build_mode,
                lit_dir_rust_manifest_str,
                get_num_jobs_cargo(),
            ))
            .output()
            .expect("Failed building downstream rust project for lit tests");
        let stderr_cargo: &[u8] = output_cargo.stderr.as_slice();
        let stdout_cargo: &[u8] = output_cargo.stdout.as_slice();

        println!();
        eprintln!("Cargo build '{}':\n{}",
            lit_dir_rust_tests_dir_str,
            std::str::from_utf8(stderr_cargo).unwrap(),
        );
        println!("Cargo build '{}':\n{}",
            lit_dir_rust_tests_dir_str,
            std::str::from_utf8(stdout_cargo).unwrap(),
        );

        println!("Processing tests in directories: {}, {}",
            pathbuf_to_string(&lit_dir_mlir),
            pathbuf_to_string(&lit_dir_rust),
        );
        let output = Command::new(&shell)
            .arg("-c")
            .arg(format!("{} {} {} {}",
                lit_bin_str,
                lit_args,
                lit_dir_mlir_str,
                lit_dir_rust_tests_dir_str,
            ))
            .output()
            .expect("Failed lit tests");
        let stderr: &[u8] = output.stderr.as_slice();
        let stdout: &[u8] = output.stdout.as_slice();

        println!();
        eprintln!("Lit stderr:\n{}", std::str::from_utf8(stderr).unwrap());
        println!("Lit stdout:\n{}", std::str::from_utf8(stdout).unwrap());

        assert!(output.status.success());
    }
}
