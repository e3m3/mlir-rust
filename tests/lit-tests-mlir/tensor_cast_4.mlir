// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<?x?xf32>) -> tensor<*xf32>
    {
        %t0 = tensor.cast %t : tensor<?x?xf32> to tensor<*xf32>
        func.return %t0 : tensor<*xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<?x?xf32>) -> tensor<*xf32> {
// CHECK_CAN:           %cast = tensor.cast %arg0 : tensor<?x?xf32> to tensor<*xf32>
// CHECK_CAN:           return %cast : tensor<*xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<?x?xf32>) -> tensor<*xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<?x?xf32>):
// CHECK_GEN:           %0 = "tensor.cast"(%arg0) : (tensor<?x?xf32>) -> tensor<*xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
