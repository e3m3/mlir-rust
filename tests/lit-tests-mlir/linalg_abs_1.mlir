// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<8x16xf32>) -> tensor<8x16xf32>
    {
        %out = tensor.empty() : tensor<8x16xf32>
        %out0 = linalg.abs ins(%a: tensor<8x16xf32>) outs(%out: tensor<8x16xf32>) -> tensor<8x16xf32>
        func.return %out0 : tensor<8x16xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<8x16xf32>
// CHECK_CAN:           %1 = linalg.abs ins(%arg0 : tensor<8x16xf32>) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK_CAN:           return %1 : tensor<8x16xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<8x16xf32>) -> tensor<8x16xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<8x16xf32>):
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<8x16xf32>
// CHECK_GEN:           %1 = "linalg.abs"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK_GEN:           ^bb0(%arg1: f32, %arg2: f32):
// CHECK_GEN:               %2 = "math.absf"(%arg1) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK_GEN:               "linalg.yield"(%2) : (f32) -> ()
// CHECK_GEN:           }) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK_GEN:           "func.return"(%1) : (tensor<8x16xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
