// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<10xi32>, %b: tensor<10xi32>) -> tensor<10xi32>
    {
        %out = arith.divsi %a, %b : tensor<10xi32>
        func.return %out : tensor<10xi32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
// CHECK_CAN:           %0 = arith.divsi %arg0, %arg1 : tensor<10xi32>
// CHECK_CAN:           return %0 : tensor<10xi32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>):
// CHECK_GEN:           %0 = "arith.divsi"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
// CHECK_GEN:           "func.return"(%0) : (tensor<10xi32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
