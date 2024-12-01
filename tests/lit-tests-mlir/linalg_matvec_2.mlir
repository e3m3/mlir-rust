// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  tensor

module {
    func.func @test(%a: memref<8x16xf32>, %b: memref<16xf32>) -> memref<8xf32>
    {
        %out = memref.alloc() {alignment = 8} : memref<8xf32>
        linalg.matvec ins(%a, %b: memref<8x16xf32>, memref<16xf32>) outs(%out: memref<8xf32>)
        func.return %out : memref<8xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<8x16xf32>, %arg1: memref<16xf32>) -> memref<8xf32> {
// CHECK_CAN:           %alloc = memref.alloc() {alignment = 8 : i64} : memref<8xf32>
// CHECK_CAN:           linalg.matvec ins(%arg0, %arg1 : memref<8x16xf32>, memref<16xf32>) outs(%alloc : memref<8xf32>)
// CHECK_CAN:           return %alloc : memref<8xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK_GEN:   #map2 = affine_map<(d0, d1) -> (d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<8x16xf32>, memref<16xf32>) -> memref<8xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<8x16xf32>, %arg1: memref<16xf32>):
// CHECK_GEN:           %0 = "memref.alloc"() <{alignment = 8 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xf32>
// CHECK_GEN:           "linalg.matvec"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
// CHECK_GEN:               %1 = "arith.mulf"(%arg2, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               %2 = "arith.addf"(%arg4, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               "linalg.yield"(%2) : (f32) -> ()
// CHECK_GEN:           }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (memref<8x16xf32>, memref<16xf32>, memref<8xf32>) -> ()
// CHECK_GEN:           "func.return"(%0) : (memref<8xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
