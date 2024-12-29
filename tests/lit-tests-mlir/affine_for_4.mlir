// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func private @black_box(index) -> ()
    func.func @test(%n: index, %N: index, %m: index, %M: index) -> ()
    {
        affine.for %i =
            max affine_map<()[s0,s1] -> (s0,s1,10)> ()[%n,%m] to 
            min affine_map<()[s0,s1] -> (s0,s1,100)> ()[%N,%M] step 1
        {
            func.call @black_box(%i) : (index) -> ()
        }
        func.return
    }
}

// CHECK_CAN:   #map = affine_map<()[s0, s1] -> (s0, s1, 10)>
// CHECK_CAN:   #map1 = affine_map<()[s0, s1] -> (s0, s1, 100)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index)
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
// CHECK_CAN:           affine.for %arg4 = max #map()[%arg0, %arg2] to min #map1()[%arg1, %arg3] {
// CHECK_CAN:               func.call @black_box(%arg4) : (index) -> ()
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<()[s0, s1] -> (s0, s1, 10)>
// CHECK_GEN:   #map1 = affine_map<()[s0, s1] -> (s0, s1, 100)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
// CHECK_GEN:           "affine.for"(%arg0, %arg2, %arg1, %arg3) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 2, 2, 0>, step = 1 : index, upperBoundMap = #map1}> ({
// CHECK_GEN:           ^bb0(%arg4: index):
// CHECK_GEN:               "func.call"(%arg4) <{callee = @black_box}> : (index) -> ()
// CHECK_GEN:               "affine.yield"() : () -> ()
// CHECK_GEN:           }) : (index, index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
