// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(
        %base: memref<?x?x?x?xf32>,
        %i: index,
        %j: index,
        %k: index,
        %l: index,
        %value: vector<16x32x64xf32>
    ) -> () {
        vector.transfer_write %value, %base[%i, %j, %k, %l] {
            permutation_map = affine_map<(d0,d1,d2,d3) -> (d3,d1,d2)>
        } : vector<16x32x64xf32>, memref<?x?x?x?xf32>
        func.return
    }
}

// CHECK_CAN:   #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: vector<16x32x64xf32>) {
// CHECK_CAN:           vector.transfer_write %arg5, %arg0[%arg1, %arg2, %arg3, %arg4] {permutation_map = #map} : vector<16x32x64xf32>, memref<?x?x?x?xf32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?x?x?xf32>, index, index, index, index, vector<16x32x64xf32>) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: vector<16x32x64xf32>):
// CHECK_GEN:           "vector.transfer_write"(%arg5, %arg0, %arg1, %arg2, %arg3, %arg4) <{in_bounds = [false, false, false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = #map}> : (vector<16x32x64xf32>, memref<?x?x?x?xf32>, index, index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
