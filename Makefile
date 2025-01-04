.PHONY: all container format format-check native clean clean-native

CONTAINER_BIN=podman
CONTAINER_FILE=container/Containerfile
CONTAINER_NAME=mlir-rust

TESTS_DIR=tests/lit-tests-rust
CARGO_TOML=Cargo.toml
CARGO_TOML_TESTS=${TESTS_DIR}/Cargo.toml

all: container

container: 
	${CONTAINER_BIN} build -t ${CONTAINER_NAME} -f ${CONTAINER_FILE} .

clean:
	${CONTAINER_BIN} image rm ${CONTAINER_NAME}

native:
	cargo build

clean-native:
	cargo clean

format:
	cargo fmt --all --manifest-path ${CARGO_TOML}
	cargo fmt --all --manifest-path ${CARGO_TOML_TESTS}
	cargo fmt --all --manifest-path ${CARGO_TOML_TESTS} -- ${TESTS_DIR}/src/*.lit-rs

format-check:
	cargo fmt --all --check --manifest-path ${CARGO_TOML} && \
	cargo fmt --all --check --manifest-path ${CARGO_TOML_TESTS} && \
	cargo fmt --all --check --manifest-path ${CARGO_TOML_TESTS} -- ${TESTS_DIR}/src/*.lit-rs
