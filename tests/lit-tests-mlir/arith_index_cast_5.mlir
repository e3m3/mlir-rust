// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%in: tensor<10xi32>) -> tensor<10xindex>
    {
        %out = arith.index_cast %in : tensor<10xi32> to tensor<10xindex>
        func.return %out : tensor<10xindex>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xi32>) -> tensor<10xindex> {
// CHECK_CAN:           %0 = arith.index_cast %arg0 : tensor<10xi32> to tensor<10xindex>
// CHECK_CAN:           return %0 : tensor<10xindex>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xi32>) -> tensor<10xindex>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xi32>):
// CHECK_GEN:           %0 = "arith.index_cast"(%arg0) : (tensor<10xi32>) -> tensor<10xindex>
// CHECK_GEN:           "func.return"(%0) : (tensor<10xindex>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
