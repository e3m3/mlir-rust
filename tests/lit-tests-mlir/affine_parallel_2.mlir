// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func private @black_box(index, index) -> ()
    func.func @test(%N: index, %M: index) -> ()
    {
        affine.parallel (%ii, %jj) = (0, 0) to (%N, %M) step (32, 32) {
            affine.parallel (%i, %j) = (%ii, %jj) to (min(%ii + 32, %N), min(%jj + 32, %M)) {
                func.call @black_box(%i, %j) : (index, index) -> ()
            }
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index, index)
// CHECK_CAN:           func.func @test(%arg0: index, %arg1: index) {
// CHECK_CAN:           affine.parallel (%arg2, %arg3) = (0, 0) to (symbol(%arg0), symbol(%arg1)) step (32, 32) {
// CHECK_CAN:               affine.parallel (%arg4, %arg5) = (%arg2, %arg3) to (min(%arg2 + 32, symbol(%arg0)), min(%arg3 + 32, symbol(%arg1))) {
// CHECK_CAN:                   func.call @black_box(%arg4, %arg5) : (index, index) -> ()
// CHECK_CAN:               }
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1, d2, d3) -> (d0 + 32, d1, d2 + 32, d3)>
// CHECK_GEN:   #map2 = affine_map<() -> (0, 0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           "affine.parallel"(%arg0, %arg1) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map2, reductions = [], steps = [32, 32], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map}> ({
// CHECK_GEN:           ^bb0(%arg2: index, %arg3: index):
// CHECK_GEN:               "affine.parallel"(%arg2, %arg3, %arg2, %arg0, %arg3, %arg1) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map, reductions = [], steps = [1, 1], upperBoundsGroups = dense<2> : tensor<2xi32>, upperBoundsMap = #map1}> ({
// CHECK_GEN:               ^bb0(%arg4: index, %arg5: index):
// CHECK_GEN:                   "func.call"(%arg4, %arg5) <{callee = @black_box}> : (index, index) -> ()
// CHECK_GEN:                   "affine.yield"() : () -> ()
// CHECK_GEN:               }) : (index, index, index, index, index, index) -> ()
// CHECK_GEN:               "affine.yield"() : () -> ()
// CHECK_GEN:           }) : (index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
