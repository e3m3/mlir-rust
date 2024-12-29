// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%c1: index, %c2: index, %init0: index, %init1: index) -> (index, index) {
        %out:2 = affine.for %i = 5 to 128 step 1 iter_args(%acc0 = %init0, %acc1 = %init1)
            -> (index, index)
        {
            %sum0 = arith.addi %acc0, %c1 : index
            %sum1 = arith.addi %acc1, %c2 : index
            affine.yield %sum0, %sum1 : index, index
        }
        func.return %out#0, %out#1 : index, index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index) {
// CHECK_CAN:           %0:2 = affine.for %arg4 = 5 to 128 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (index, index) {
// CHECK_CAN:               %1 = arith.addi %arg5, %arg0 : index
// CHECK_CAN:               %2 = arith.addi %arg6, %arg1 : index
// CHECK_CAN:               affine.yield %1, %2 : index, index
// CHECK_CAN:           }
// CHECK_CAN:           return %0#0, %0#1 : index, index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<() -> (5)>
// CHECK_GEN:   #map1 = affine_map<() -> (128)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, index) -> (index, index), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
// CHECK_GEN:           %0:2 = "affine.for"(%arg2, %arg3) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = #map1}> ({
// CHECK_GEN:           ^bb0(%arg4: index, %arg5: index, %arg6: index):
// CHECK_GEN:               %1 = "arith.addi"(%arg5, %arg0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK_GEN:               %2 = "arith.addi"(%arg6, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK_GEN:               "affine.yield"(%1, %2) : (index, index) -> ()
// CHECK_GEN:           }) : (index, index) -> (index, index)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (index, index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
