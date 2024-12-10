// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<4x4xi32>, %i: index, %j: index) -> i32
    {
        %out = tensor.extract %t[%i, %j] : tensor<4x4xi32>
        func.return %out : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<4x4xi32>, %arg1: index, %arg2: index) -> i32 {
// CHECK_CAN:           %extracted = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xi32>
// CHECK_CAN:           return %extracted : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<4x4xi32>, index, index) -> i32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<4x4xi32>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "tensor.extract"(%arg0, %arg1, %arg2) : (tensor<4x4xi32>, index, index) -> i32
// CHECK_GEN:           "func.return"(%0) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
