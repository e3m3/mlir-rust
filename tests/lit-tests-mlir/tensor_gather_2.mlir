// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<3x4x5xf32>, %indices: tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32>
    {
        %out = tensor.gather %t[%indices] gather_dims([1]) :
            (tensor<3x4x5xf32>, tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32>
        func.return %out : tensor<6x7x3x1x5xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<3x4x5xf32>, %arg1: tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32> {
// CHECK_CAN:           %gather = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<3x4x5xf32>, tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32>
// CHECK_CAN:           return %gather : tensor<6x7x3x1x5xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<3x4x5xf32>, tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<3x4x5xf32>, %arg1: tensor<6x7x1xindex>):
// CHECK_GEN:           %0 = "tensor.gather"(%arg0, %arg1) <{gather_dims = array<i64: 1>}> : (tensor<3x4x5xf32>, tensor<6x7x1xindex>) -> tensor<6x7x3x1x5xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<6x7x3x1x5xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
