// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<8x16x4xf32>) -> tensor<16x4xf32>
    {
        %out = tensor.extract_slice %t[0,0,0][1,16,4][1,1,1] : tensor<8x16x4xf32> to tensor<16x4xf32>
        func.return %out : tensor<16x4xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<8x16x4xf32>) -> tensor<16x4xf32> {
// CHECK_CAN:           %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 16, 4] [1, 1, 1] : tensor<8x16x4xf32> to tensor<16x4xf32>
// CHECK_CAN:           return %extracted_slice : tensor<16x4xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<8x16x4xf32>) -> tensor<16x4xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<8x16x4xf32>):
// CHECK_GEN:           %0 = "tensor.extract_slice"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 16, 4>, static_strides = array<i64: 1, 1, 1>}> : (tensor<8x16x4xf32>) -> tensor<16x4xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<16x4xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
