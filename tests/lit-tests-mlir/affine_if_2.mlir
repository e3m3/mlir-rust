// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%A: memref<10x10xf32>, %i: index, %j: index) -> f32
    {
        %out = affine.if affine_set<(i,j): (
            i - 1 >= 0, 10 - i >= 0,
            j - 1 >= 0, 10 - j >= 0
        )> (%i, %j) -> f32 {
            %tmp = affine.load %A[%i - 1, %j - 1] : memref<10x10xf32>
            affine.yield %tmp : f32
        } else {
            %tmp = arith.constant 0.0 : f32
            affine.yield %tmp : f32
        }
        func.return %out : f32
    }
}

// CHECK_CAN:   #set = affine_set<()[s0, s1] : (s0 - 1 >= 0, -s0 + 10 >= 0, s1 - 1 >= 0, -s1 + 10 >= 0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<10x10xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK_CAN:           %cst = arith.constant 0.000000e+00 : f32
// CHECK_CAN:           %0 = affine.if #set()[%arg1, %arg2] -> f32 {
// CHECK_CAN:               %1 = affine.load %arg0[symbol(%arg1) - 1, symbol(%arg2) - 1] : memref<10x10xf32>
// CHECK_CAN:               affine.yield %1 : f32
// CHECK_CAN:           } else {
// CHECK_CAN:               affine.yield %cst : f32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0 - 1, d1 - 1)>
// CHECK_GEN:   #set = affine_set<(d0, d1) : (d0 - 1 >= 0, -d0 + 10 >= 0, d1 - 1 >= 0, -d1 + 10 >= 0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<10x10xf32>, index, index) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<10x10xf32>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "affine.if"(%arg1, %arg2) ({
// CHECK_GEN:               %2 = "affine.load"(%arg0, %arg1, %arg2) <{map = #map}> : (memref<10x10xf32>, index, index) -> f32
// CHECK_GEN:               "affine.yield"(%2) : (f32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CHECK_GEN:               "affine.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }) {condition = #set} : (index, index) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
