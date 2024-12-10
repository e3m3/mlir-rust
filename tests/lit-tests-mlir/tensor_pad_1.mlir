// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<10xi32>, %padding: i32) -> tensor<18xi32>
    {
        %out = tensor.pad %t low[3] high[5] {
        ^bb0(%i: index):
            tensor.yield %padding : i32
        } : tensor<10xi32> to tensor<18xi32>
        func.return %out : tensor<18xi32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<10xi32>, %arg1: i32) -> tensor<18xi32> {
// CHECK_CAN:           %padded = tensor.pad %arg0 low[3] high[5] {
// CHECK_CAN:           ^bb0(%arg2: index):
// CHECK_CAN:               tensor.yield %arg1 : i32
// CHECK_CAN:           } : tensor<10xi32> to tensor<18xi32>
// CHECK_CAN:           return %padded : tensor<18xi32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<10xi32>, i32) -> tensor<18xi32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<10xi32>, %arg1: i32):
// CHECK_GEN:           %0 = "tensor.pad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_high = array<i64: 5>, static_low = array<i64: 3>}> ({
// CHECK_GEN:           ^bb0(%arg2: index):
// CHECK_GEN:               "tensor.yield"(%arg1) : (i32) -> ()
// CHECK_GEN:           }) : (tensor<10xi32>) -> tensor<18xi32>
// CHECK_GEN:           "func.return"(%0) : (tensor<18xi32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
