// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi64>
    {
        %out = arith.ceildivsi %a, %b : tensor<4xi64>
        func.return %out : tensor<4xi64>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi64> {
// CHECK_CAN:           %0 = arith.ceildivsi %arg0, %arg1 : tensor<4xi64>
// CHECK_CAN:           return %0 : tensor<4xi64>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>):
// CHECK_GEN:           %0 = "arith.ceildivsi"(%arg0, %arg1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
// CHECK_GEN:           "func.return"(%0) : (tensor<4xi64>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
