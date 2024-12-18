// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<4x4x4xf32>, %indices: tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
    {
        %out = tensor.gather %t[%indices] gather_dims([0,1,2]) {unique} :
            (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
        func.return %out : tensor<1x2x1x1x1xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<4x4x4xf32>, %arg1: tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32> {
// CHECK_CAN:           %gather = tensor.gather %arg0[%arg1] gather_dims([0, 1, 2]) unique : (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
// CHECK_CAN:           return %gather : tensor<1x2x1x1x1xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<4x4x4xf32>, %arg1: tensor<1x2x3xindex>):
// CHECK_GEN:           %0 = "tensor.gather"(%arg0, %arg1) <{gather_dims = array<i64: 0, 1, 2>, unique}> : (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<1x2x1x1x1xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
