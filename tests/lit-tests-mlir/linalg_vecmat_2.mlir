// RUN: @mlir-opt -h                                    | @filecheck %s
// RUN: @mlir-opt %s --canonicalize                     | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic            | @filecheck %s --check-prefix=CHECK_GEN
// RUN: @mlir-opt %s --convert-linalg-to-affine-loops   | @filecheck %s --check-prefix=CHECK_AFF

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  memref

module {
    func.func @test(%a: memref<16xf32>, %b: memref<16x8xf32>) -> memref<8xf32>
    {
        %out = memref.alloc() : memref<8xf32>
        linalg.vecmat ins(%a, %b: memref<16xf32>, memref<16x8xf32>) outs(%out: memref<8xf32>)
        func.return %out : memref<8xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<16xf32>, %arg1: memref<16x8xf32>) -> memref<8xf32> {
// CHECK_CAN:           %alloc = memref.alloc() : memref<8xf32>
// CHECK_CAN:           linalg.vecmat ins(%arg0, %arg1 : memref<16xf32>, memref<16x8xf32>) outs(%alloc : memref<8xf32>)
// CHECK_CAN:           return %alloc : memref<8xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d1)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1) -> (d1, d0)>
// CHECK_GEN:   #map2 = affine_map<(d0, d1) -> (d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<16xf32>, memref<16x8xf32>) -> memref<8xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<16xf32>, %arg1: memref<16x8xf32>):
// CHECK_GEN:           %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xf32>
// CHECK_GEN:           "linalg.vecmat"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
// CHECK_GEN:               %1 = "arith.mulf"(%arg2, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               %2 = "arith.addf"(%arg4, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               "linalg.yield"(%2) : (f32) -> ()
// CHECK_GEN:           }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (memref<16xf32>, memref<16x8xf32>, memref<8xf32>) -> ()
// CHECK_GEN:           "func.return"(%0) : (memref<8xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()

// CHECK_AFF:   module {
// CHECK_AFF:       func.func @test(%arg0: memref<16xf32>, %arg1: memref<16x8xf32>) -> memref<8xf32> {
// CHECK_AFF:           %alloc = memref.alloc() : memref<8xf32>
// CHECK_AFF:           affine.for %arg2 = 0 to 8 {
// CHECK_AFF:               affine.for %arg3 = 0 to 16 {
// CHECK_AFF:                   %0 = affine.load %arg0[%arg3] : memref<16xf32>
// CHECK_AFF:                   %1 = affine.load %arg1[%arg3, %arg2] : memref<16x8xf32>
// CHECK_AFF:                   %2 = affine.load %alloc[%arg2] : memref<8xf32>
// CHECK_AFF:                   %3 = arith.mulf %0, %1 : f32
// CHECK_AFF:                   %4 = arith.addf %2, %3 : f32
// CHECK_AFF:                   affine.store %4, %alloc[%arg2] : memref<8xf32>
// CHECK_AFF:               }
// CHECK_AFF:           }
// CHECK_AFF:           return %alloc : memref<8xf32>
// CHECK_AFF:       }
// CHECK_AFF:   }
