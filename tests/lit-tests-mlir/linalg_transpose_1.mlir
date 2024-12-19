// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<16x64xf32>) -> tensor<64x16xf32>
    {
        %out = tensor.empty() : tensor<64x16xf32>
        %out0 = linalg.transpose ins(%a: tensor<16x64xf32>) outs(%out: tensor<64x16xf32>)
            permutation = [1, 0]
        func.return %out0 : tensor<64x16xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<16x64xf32>) -> tensor<64x16xf32> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<64x16xf32>
// CHECK_CAN:           %transposed = linalg.transpose ins(%arg0 : tensor<16x64xf32>) outs(%0 : tensor<64x16xf32>) permutation = [1, 0]
// CHECK_CAN:           return %transposed : tensor<64x16xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<16x64xf32>) -> tensor<64x16xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<16x64xf32>):
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<64x16xf32>
// CHECK_GEN:           %1 = "linalg.transpose"(%arg0, %0) <{permutation = array<i64: 1, 0>}> ({
// CHECK_GEN:           ^bb0(%arg1: f32, %arg2: f32):
// CHECK_GEN:               "linalg.yield"(%arg1) : (f32) -> ()
// CHECK_GEN:           }) : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<64x16xf32>
// CHECK_GEN:           "func.return"(%1) : (tensor<64x16xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
