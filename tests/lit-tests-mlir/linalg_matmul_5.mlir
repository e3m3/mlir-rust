// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: tensor<3x5xi64>, %b: tensor<5x7xi64>) -> tensor<3x7xi64>
    {
        %out = tensor.empty() : tensor<3x7xi64>
        %out0 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
            ins(%a, %b: tensor<3x5xi64>, tensor<5x7xi64>) outs(%out: tensor<3x7xi64>) -> tensor<3x7xi64>
        func.return %out0 : tensor<3x7xi64>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<3x5xi64>, %arg1: tensor<5x7xi64>) -> tensor<3x7xi64> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<3x7xi64>
// CHECK_CAN:           %1 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(%arg0, %arg1 : tensor<3x5xi64>, tensor<5x7xi64>) outs(%0 : tensor<3x7xi64>) -> tensor<3x7xi64>
// CHECK_CAN:           return %1 : tensor<3x7xi64>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK_GEN:   #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<3x5xi64>, tensor<5x7xi64>) -> tensor<3x7xi64>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<3x5xi64>, %arg1: tensor<5x7xi64>):
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<3x7xi64>
// CHECK_GEN:           %1 = "linalg.matmul"(%arg0, %arg1, %0) <{cast = #linalg.type_fn<cast_unsigned>, operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
// CHECK_GEN:               %2 = "arith.muli"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK_GEN:               %3 = "arith.addi"(%arg4, %2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK_GEN:               "linalg.yield"(%3) : (i64) -> ()
// CHECK_GEN:           }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<3x5xi64>, tensor<5x7xi64>, tensor<3x7xi64>) -> tensor<3x7xi64>
// CHECK_GEN:           "func.return"(%1) : (tensor<3x7xi64>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
