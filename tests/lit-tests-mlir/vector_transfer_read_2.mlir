// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(
        %base: memref<?x?x?xf32>,
        %i: index,
        %j: index,
        %k: index,
        %default: f32
    ) -> vector<32x256xf32> {
        %v = vector.transfer_read %base[%i, %j, %k], %default {
            permutation_map = affine_map<(d0,d1,d2) -> (d2,d1)>
        } : memref<?x?x?xf32>, vector<32x256xf32>
        func.return %v : vector<32x256xf32>
    }
}

// CHECK_CAN:   #map = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> vector<32x256xf32> {
// CHECK_CAN:           %0 = vector.transfer_read %arg0[%arg1, %arg2, %arg3], %arg4 {permutation_map = #map} : memref<?x?x?xf32>, vector<32x256xf32>
// CHECK_CAN:           return %0 : vector<32x256xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?x?xf32>, index, index, index, f32) -> vector<32x256xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: f32):
// CHECK_GEN:           %0 = "vector.transfer_read"(%arg0, %arg1, %arg2, %arg3, %arg4) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = #map}> : (memref<?x?x?xf32>, index, index, index, f32) -> vector<32x256xf32>
// CHECK_GEN:           "func.return"(%0) : (vector<32x256xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
