// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test(%n: index, %N: index) -> i32
    {
        %init = arith.constant 0 : i32
        %out = affine.for %i = %n to %N step 2 iter_args(%acc = %init) -> i32 {
            %value = index.casts %i : index to i32
            %sum = arith.addi %acc, %value : i32
            affine.yield %sum : i32
        }
        func.return %out : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> i32 {
// CHECK_CAN:           %c0_i32 = arith.constant 0 : i32
// CHECK_CAN:           %0 = affine.for %arg2 = %arg0 to %arg1 step 2 iter_args(%arg3 = %c0_i32) -> (i32) {
// CHECK_CAN:               %1 = index.casts %arg2 : index to i32
// CHECK_CAN:               %2 = arith.addi %arg3, %1 : i32
// CHECK_CAN:               affine.yield %2 : i32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<()[s0] -> (s0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> i32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK_GEN:           %1 = "affine.for"(%arg0, %arg1, %0) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 1, 1, 1>, step = 2 : index, upperBoundMap = #map}> ({
// CHECK_GEN:           ^bb0(%arg2: index, %arg3: i32):
// CHECK_GEN:               %2 = "index.casts"(%arg2) : (index) -> i32
// CHECK_GEN:               %3 = "arith.addi"(%arg3, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK_GEN:               "affine.yield"(%3) : (i32) -> ()
// CHECK_GEN:           }) : (index, index, i32) -> i32
// CHECK_GEN:           "func.return"(%1) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
