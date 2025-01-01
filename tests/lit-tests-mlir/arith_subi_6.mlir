// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<10xindex>, %b: tensor<10xindex>) -> tensor<10xindex>
    {
        %out = arith.subi %a, %b {overflowFlags = #arith.overflow<nsw>} : tensor<10xindex>
        func.return %out : tensor<10xindex>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xindex>, %arg1: tensor<10xindex>) -> tensor<10xindex> {
// CHECK_CAN:           %0 = arith.subi %arg0, %arg1 overflow<nsw> : tensor<10xindex>
// CHECK_CAN:           return %0 : tensor<10xindex>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xindex>, tensor<10xindex>) -> tensor<10xindex>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xindex>, %arg1: tensor<10xindex>):
// CHECK_GEN:           %0 = "arith.subi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<nsw>}> : (tensor<10xindex>, tensor<10xindex>) -> tensor<10xindex>
// CHECK_GEN:           "func.return"(%0) : (tensor<10xindex>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
