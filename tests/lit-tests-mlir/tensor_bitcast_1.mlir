// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<4xui32>) -> tensor<4xi32>
    {
        %t0 = tensor.bitcast %t : tensor<4xui32> to tensor<4xi32>
        func.return %t0 : tensor<4xi32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<4xui32>) -> tensor<4xi32> {
// CHECK_CAN:           %0 = tensor.bitcast %arg0 : tensor<4xui32> to tensor<4xi32>
// CHECK_CAN:           return %0 : tensor<4xi32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<4xui32>) -> tensor<4xi32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<4xui32>):
// CHECK_GEN:           %0 = "tensor.bitcast"(%arg0) : (tensor<4xui32>) -> tensor<4xi32>
// CHECK_GEN:           "func.return"(%0) : (tensor<4xi32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
