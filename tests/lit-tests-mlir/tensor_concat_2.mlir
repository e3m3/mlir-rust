// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t0: tensor<3x?xf32>, %t1: tensor<3x2xf32>, %t2: tensor<3x?xf32>) -> tensor<3x10xf32>
    {
        %out = tensor.concat dim(1) %t0, %t1, %t2 :
            (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32>) -> tensor<3x10xf32>
        func.return %out : tensor<3x10xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<3x?xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x?xf32>) -> tensor<3x10xf32> {
// CHECK_CAN:           %concat = tensor.concat dim(1) %arg0, %arg1, %arg2 : (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32>) -> tensor<3x10xf32>
// CHECK_CAN:           return %concat : tensor<3x10xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32>) -> tensor<3x10xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<3x?xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x?xf32>):
// CHECK_GEN:           %0 = "tensor.concat"(%arg0, %arg1, %arg2) <{dim = 1 : i64}> : (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32>) -> tensor<3x10xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<3x10xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
