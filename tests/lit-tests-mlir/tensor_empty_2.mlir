// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%m: index, %n: index) -> tensor<?x?xf32>
    {
        %out = tensor.empty(%m, %n) : tensor<?x?xf32>
        func.return %out : tensor<?x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> tensor<?x?xf32> {
// CHECK_CAN:           %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
// CHECK_CAN:           return %0 : tensor<?x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> tensor<?x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "tensor.empty"(%arg0, %arg1) : (index, index) -> tensor<?x?xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<?x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
