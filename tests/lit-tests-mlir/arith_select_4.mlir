// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%cond: i1, %a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32>
    {
        %out = arith.select %cond, %a, %b : tensor<4xi32>
        func.return %out : tensor<4xi32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<4xi32> {
// CHECK_CAN:           %0 = arith.select %arg0, %arg1, %arg2 : tensor<4xi32>
// CHECK_CAN:           return %0 : tensor<4xi32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i1, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>):
// CHECK_GEN:           %0 = "arith.select"(%arg0, %arg1, %arg2) : (i1, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK_GEN:           "func.return"(%0) : (tensor<4xi32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
