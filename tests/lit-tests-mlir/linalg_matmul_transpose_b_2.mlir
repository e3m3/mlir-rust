// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<3x5xf32>, %b: tensor<7x5xf32>) -> tensor<3x7xf32>
    {
        %out = tensor.empty() : tensor<3x7xf32>
        %out0 = linalg.matmul_transpose_b
            ins(%a, %b: tensor<3x5xf32>, tensor<7x5xf32>) outs(%out: tensor<3x7xf32>) -> tensor<3x7xf32>
        func.return %out0 : tensor<3x7xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<3x5xf32>, %arg1: tensor<7x5xf32>) -> tensor<3x7xf32> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<3x7xf32>
// CHECK_CAN:           %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<3x5xf32>, tensor<7x5xf32>) outs(%0 : tensor<3x7xf32>) -> tensor<3x7xf32>
// CHECK_CAN:           return %1 : tensor<3x7xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK_GEN:   #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<3x5xf32>, tensor<7x5xf32>) -> tensor<3x7xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<3x5xf32>, %arg1: tensor<7x5xf32>):
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<3x7xf32>
// CHECK_GEN:           %1 = "linalg.matmul_transpose_b"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
// CHECK_GEN:               %2 = "arith.mulf"(%arg2, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               %3 = "arith.addf"(%arg4, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               "linalg.yield"(%3) : (f32) -> ()
// CHECK_GEN:           }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<3x5xf32>, tensor<7x5xf32>, tensor<3x7xf32>) -> tensor<3x7xf32>
// CHECK_GEN:           "func.return"(%1) : (tensor<3x7xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
