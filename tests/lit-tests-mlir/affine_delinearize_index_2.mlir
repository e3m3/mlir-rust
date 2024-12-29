// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// COM: This test uses inline constant values for the basis, but is not accepted by `mlir-opt`
// COM: as expected from the documentation [1].
// COM: [1]: https://mlir.llvm.org/docs/Dialects/Affine/#affinedelinearize_index-affineaffinedelinearizeindexop
// XFAIL: *

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test() -> (index, index, index)
    {
        %i = index.constant 2048
        %out:3 = affine.delinearize_index %i into (244, 244) : index, index
        func.return %out#0, %out#1, %out#2 : index, index, index
    }
}

// CHECK_CAN:   See COM.

// CHECK_GEN:   See COM.
