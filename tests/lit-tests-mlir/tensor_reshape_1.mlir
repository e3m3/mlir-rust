// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<4x1xf32>, %shape: tensor<1xi32>) -> tensor<4xf32>
    {
        %out = tensor.reshape %t(%shape) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
        func.return %out : tensor<4xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<4x1xf32>, %arg1: tensor<1xi32>) -> tensor<4xf32> {
// CHECK_CAN:           %reshape = tensor.reshape %arg0(%arg1) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK_CAN:           return %reshape : tensor<4xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<4x1xf32>, %arg1: tensor<1xi32>):
// CHECK_GEN:           %0 = "tensor.reshape"(%arg0, %arg1) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<4xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
