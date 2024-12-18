// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<8x16xi64>, %b: tensor<8x16xi64>) -> tensor<8x16xi64>
    {
        %out = tensor.empty() : tensor<8x16xi64>
        %out0 = linalg.mul ins(%a, %b: tensor<8x16xi64>, tensor<8x16xi64>) outs(%out: tensor<8x16xi64>) -> tensor<8x16xi64>
        func.return %out0 : tensor<8x16xi64>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<8x16xi64>, %arg1: tensor<8x16xi64>) -> tensor<8x16xi64> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<8x16xi64>
// CHECK_CAN:           %1 = linalg.mul ins(%arg0, %arg1 : tensor<8x16xi64>, tensor<8x16xi64>) outs(%0 : tensor<8x16xi64>) -> tensor<8x16xi64>
// CHECK_CAN:           return %1 : tensor<8x16xi64>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi64>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<8x16xi64>, %arg1: tensor<8x16xi64>):
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<8x16xi64>
// CHECK_GEN:           %1 = "linalg.mul"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
// CHECK_GEN:               %2 = "arith.muli"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK_GEN:               "linalg.yield"(%2) : (i64) -> ()
// CHECK_GEN:           }) : (tensor<8x16xi64>, tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi64>
// CHECK_GEN:           "func.return"(%1) : (tensor<8x16xi64>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
