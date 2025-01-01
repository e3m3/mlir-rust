// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<10xtf32>, %b: tensor<10xtf32>) -> tensor<10xtf32>
    {
        %out = arith.mulf %a, %b {fastmath = #arith.fastmath<fast>} : tensor<10xtf32>
        func.return %out : tensor<10xtf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xtf32>, %arg1: tensor<10xtf32>) -> tensor<10xtf32> {
// CHECK_CAN:           %0 = arith.mulf %arg0, %arg1 fastmath<fast> : tensor<10xtf32>
// CHECK_CAN:           return %0 : tensor<10xtf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xtf32>, tensor<10xtf32>) -> tensor<10xtf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xtf32>, %arg1: tensor<10xtf32>):
// CHECK_GEN:           %0 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<fast>}> : (tensor<10xtf32>, tensor<10xtf32>) -> tensor<10xtf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<10xtf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
