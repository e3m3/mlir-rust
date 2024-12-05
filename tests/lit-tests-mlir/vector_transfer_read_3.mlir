// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(
        %base: memref<?x?xf32>,
        %i: index,
        %j: index,
        %default: f32
    ) -> vector<128xf32> {
        %v = vector.transfer_read %base[%i, %j], %default {
            in_bounds = [true],
            permutation_map = affine_map<(d0,d1) -> (0)>
        } : memref<?x?xf32>, vector<128xf32>
        func.return %v : vector<128xf32>
    }
}

// CHECK_CAN:   #map = affine_map<(d0, d1) -> (0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: f32) -> vector<128xf32> {
// CHECK_CAN:           %0 = vector.transfer_read %arg0[%arg1, %arg2], %arg3 {in_bounds = [true], permutation_map = #map} : memref<?x?xf32>, vector<128xf32>
// CHECK_CAN:           return %0 : vector<128xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?xf32>, index, index, f32) -> vector<128xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: f32):
// CHECK_GEN:           %0 = "vector.transfer_read"(%arg0, %arg1, %arg2, %arg3) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = #map}> : (memref<?x?xf32>, index, index, f32) -> vector<128xf32>
// CHECK_GEN:           "func.return"(%0) : (vector<128xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
