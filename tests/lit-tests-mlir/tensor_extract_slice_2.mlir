// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(
        %t: tensor<8x16x4xf32>,
        %o0: index,
        %o2: index,
        %sz1: index,
        %st1: index
    ) -> tensor<1x?xf32> {
        %out = tensor.extract_slice %t[%o0,4,%o2][1,%sz1,1][1,%st1,1] :
            tensor<8x16x4xf32> to tensor<1x?xf32>
        func.return %out : tensor<1x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<8x16x4xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?xf32> {
// CHECK_CAN:           %extracted_slice = tensor.extract_slice %arg0[%arg1, 4, %arg2] [1, %arg3, 1] [1, %arg4, 1] : tensor<8x16x4xf32> to tensor<1x?xf32>
// CHECK_CAN:           return %extracted_slice : tensor<1x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<8x16x4xf32>, index, index, index, index) -> tensor<1x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<8x16x4xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
// CHECK_GEN:           %0 = "tensor.extract_slice"(%arg0, %arg1, %arg2, %arg3, %arg4) <{operandSegmentSizes = array<i32: 1, 2, 1, 1>, static_offsets = array<i64: -9223372036854775808, 4, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808, 1>, static_strides = array<i64: 1, -9223372036854775808, 1>}> : (tensor<8x16x4xf32>, index, index, index, index) -> tensor<1x?xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<1x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
