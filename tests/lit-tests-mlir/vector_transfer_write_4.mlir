// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor
// CHECK-SAME:  vector

module {
    func.func @test(
        %base: tensor<?x?xvector<4x3xf32>>,
        %i: index,
        %j: index,
        %value: vector<1x1x4x3xf32>
    ) -> tensor<?x?xvector<4x3xf32>> {
        %out = vector.transfer_write %value, %base[%i, %j] {
            permutation_map = affine_map<(d0,d1) -> (d0,d1)>
        } : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
        func.return %out : tensor<?x?xvector<4x3xf32>>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<?x?xvector<4x3xf32>>, %arg1: index, %arg2: index, %arg3: vector<1x1x4x3xf32>) -> tensor<?x?xvector<4x3xf32>> {
// CHECK_CAN:           %0 = vector.transfer_write %arg3, %arg0[%arg1, %arg2] : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
// CHECK_CAN:           return %0 : tensor<?x?xvector<4x3xf32>>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<?x?xvector<4x3xf32>>, index, index, vector<1x1x4x3xf32>) -> tensor<?x?xvector<4x3xf32>>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<?x?xvector<4x3xf32>>, %arg1: index, %arg2: index, %arg3: vector<1x1x4x3xf32>):
// CHECK_GEN:           %0 = "vector.transfer_write"(%arg3, %arg0, %arg1, %arg2) <{in_bounds = [false, false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = #map}> : (vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>, index, index) -> tensor<?x?xvector<4x3xf32>>
// CHECK_GEN:           "func.return"(%0) : (tensor<?x?xvector<4x3xf32>>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
