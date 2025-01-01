// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<10xi8>, %b: tensor<10xi8>) -> (tensor<10xi8>, tensor<10xi1>)
    {
        %out, %overflow = arith.addui_extended %a, %b : tensor<10xi8>, tensor<10xi1>
        func.return %out, %overflow : tensor<10xi8>, tensor<10xi1>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xi8>, %arg1: tensor<10xi8>) -> (tensor<10xi8>, tensor<10xi1>) {
// CHECK_CAN:           %sum, %overflow = arith.addui_extended %arg0, %arg1 : tensor<10xi8>, tensor<10xi1>
// CHECK_CAN:           return %sum, %overflow : tensor<10xi8>, tensor<10xi1>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xi8>, tensor<10xi8>) -> (tensor<10xi8>, tensor<10xi1>), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xi8>, %arg1: tensor<10xi8>):
// CHECK_GEN:           %0:2 = "arith.addui_extended"(%arg0, %arg1) : (tensor<10xi8>, tensor<10xi8>) -> (tensor<10xi8>, tensor<10xi1>)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (tensor<10xi8>, tensor<10xi1>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
