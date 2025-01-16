// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(
        %out: memref<98x98xf32>,
        %D: memref<100x100xf32>,
        %K: memref<3x3xf32>
    ) -> ()
    {
        affine.parallel (%x, %y) = (0, 0) to (98, 98) {
            %0 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf") -> f32 {
                %1 = affine.load %D[%x + %kx, %y + %ky] : memref<100x100xf32>
                %2 = affine.load %K[%kx, %ky] : memref<3x3xf32>
                %3 = arith.mulf %1, %2 : f32
                affine.yield %3 : f32
            }
            affine.store %0, %out[%x, %y] : memref<98x98xf32>
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<98x98xf32>, %arg1: memref<100x100xf32>, %arg2: memref<3x3xf32>) {
// CHECK_CAN:           affine.parallel (%arg3, %arg4) = (0, 0) to (98, 98) {
// CHECK_CAN:               %0 = affine.parallel (%arg5, %arg6) = (0, 0) to (2, 2) reduce ("addf") -> (f32) {
// CHECK_CAN:                   %1 = affine.load %arg1[%arg3 + %arg5, %arg4 + %arg6] : memref<100x100xf32>
// CHECK_CAN:                   %2 = affine.load %arg2[%arg5, %arg6] : memref<3x3xf32>
// CHECK_CAN:                   %3 = arith.mulf %1, %2 : f32
// CHECK_CAN:                   affine.yield %3 : f32
// CHECK_CAN:               }
// CHECK_CAN:               affine.store %0, %arg0[%arg3, %arg4] : memref<98x98xf32>
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2, d3) -> (d0 + d1, d2 + d3)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK_GEN:   #map2 = affine_map<() -> (0, 0)>
// CHECK_GEN:   #map3 = affine_map<() -> (2, 2)>
// CHECK_GEN:   #map4 = affine_map<() -> (98, 98)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<98x98xf32>, memref<100x100xf32>, memref<3x3xf32>) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<98x98xf32>, %arg1: memref<100x100xf32>, %arg2: memref<3x3xf32>):
// CHECK_GEN:           "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map2, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map4}> ({
// CHECK_GEN:           ^bb0(%arg3: index, %arg4: index):
// CHECK_GEN:               %0 = "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map2, reductions = [0], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map3}> ({
// CHECK_GEN:               ^bb0(%arg5: index, %arg6: index):
// CHECK_GEN:                   %1 = "affine.load"(%arg1, %arg3, %arg5, %arg4, %arg6) <{map = #map}> : (memref<100x100xf32>, index, index, index, index) -> f32
// CHECK_GEN:                   %2 = "affine.load"(%arg2, %arg5, %arg6) <{map = #map1}> : (memref<3x3xf32>, index, index) -> f32
// CHECK_GEN:                   %3 = "arith.mulf"(%1, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:                   "affine.yield"(%3) : (f32) -> ()
// CHECK_GEN:               }) : () -> f32
// CHECK_GEN:               "affine.store"(%0, %arg0, %arg3, %arg4) <{map = #map1}> : (f32, memref<98x98xf32>, index, index) -> ()
// CHECK_GEN:               "affine.yield"() : () -> ()
// CHECK_GEN:           }) : () -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
