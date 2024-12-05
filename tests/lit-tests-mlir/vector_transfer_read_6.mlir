// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor
// CHECK-SAME:  vector

module {
    func.func @test(%base: tensor<f32>, %default: f32) -> vector<1xf32> {
        %v = vector.transfer_read %base[], %default {
            in_bounds = [true],
            permutation_map = affine_map<() -> (0)>
        } : tensor<f32>, vector<1xf32>
        func.return %v : vector<1xf32>
    }
}

// CHECK_CAN:   #map = affine_map<() -> (0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<f32>, %arg1: f32) -> vector<1xf32> {
// CHECK_CAN:           %0 = vector.transfer_read %arg0[], %arg1 {in_bounds = [true], permutation_map = #map} : tensor<f32>, vector<1xf32>
// CHECK_CAN:           return %0 : vector<1xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<() -> (0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<f32>, f32) -> vector<1xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<f32>, %arg1: f32):
// CHECK_GEN:           %0 = "vector.transfer_read"(%arg0, %arg1) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 0, 1, 0>, permutation_map = #map}> : (tensor<f32>, f32) -> vector<1xf32>
// CHECK_GEN:           "func.return"(%0) : (vector<1xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
