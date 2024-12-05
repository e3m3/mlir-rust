// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// COM: This test fails to verify because `mlir-opt` says that this operation cannot have
// COM: broadcast dimensions.
// COM: This conflicts with the 0-d tensor example in [1].
// COM: [1]: `https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_write-vectortransferwriteop`
// XFAIL: *

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor
// CHECK-SAME:  vector

module {
    func.func @test(%base: tensor<f32>, %value: vector<1xf32>) -> tensor<f32>
    {
        %out = vector.transfer_write %value, %base[] {
            in_bounds = [true],
            permutation_map = affine_map<() -> (0)>
        } : vector<1xf32>, tensor<f32>
        func.return %out : tensor<f32>
    }
}

// CHECK_CAN:   See COM.

// CHECK_GEN:   See COM.
