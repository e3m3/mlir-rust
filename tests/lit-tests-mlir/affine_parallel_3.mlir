// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  index
// CHECK-SAME:  memref

module {
    func.func @test(%buf: memref<98x98x3xindex>, %Y: index, %Z: index, %C: index) -> ()
    {
        affine.parallel (%x, %y, %z) = (max(0, %Y), 0, max(0, %Y)) to (98, 98, min(3, %Z)) {
            %value = index.divs %z, %C
            affine.store %value, %buf[%x, %y, %z] : memref<98x98x3xindex>
        }
        return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<98x98x3xindex>, %arg1: index, %arg2: index, %arg3: index) {
// CHECK_CAN:           affine.parallel (%arg4, %arg5, %arg6) = (max(0, symbol(%arg1)), 0, max(0, symbol(%arg1))) to (98, 98, min(3, symbol(%arg2))) {
// CHECK_CAN:               %0 = index.divs %arg6, %arg3
// CHECK_CAN:               affine.store %0, %arg0[%arg4, %arg5, %arg6] : memref<98x98x3xindex>
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK_GEN:   #map1 = affine_map<(d0) -> (0, d0, 0, 0, d0)>
// CHECK_GEN:   #map2 = affine_map<(d0) -> (98, 98, 3, d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<98x98x3xindex>, index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<98x98x3xindex>, %arg1: index, %arg2: index, %arg3: index):
// CHECK_GEN:           "affine.parallel"(%arg1, %arg2) <{lowerBoundsGroups = dense<[2, 1, 2]> : tensor<3xi32>, lowerBoundsMap = #map1, reductions = [], steps = [1, 1, 1], upperBoundsGroups = dense<[1, 1, 2]> : tensor<3xi32>, upperBoundsMap = #map2}> ({
// CHECK_GEN:           ^bb0(%arg4: index, %arg5: index, %arg6: index):
// CHECK_GEN:               %0 = "index.divs"(%arg6, %arg3) : (index, index) -> index
// CHECK_GEN:               "affine.store"(%0, %arg0, %arg4, %arg5, %arg6) <{map = #map}> : (index, memref<98x98x3xindex>, index, index, index) -> ()
// CHECK_GEN:               "affine.yield"() : () -> ()
// CHECK_GEN:           }) : (index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
