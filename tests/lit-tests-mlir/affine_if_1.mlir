// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func private @black_box(index) -> ()
    func.func @test(%N: index, %i: index, %j: index) -> ()
    {
        affine.if affine_set<(d0,d1)[s0]: (
            d0 - 10 >= 0, s0 - d0 - 9 >= 0,
            d1 - 10 >= 0, s0 - d1 - 9 >= 0
        )> (%i, %j)[%N] {
            %out = affine.apply affine_map<(d0,d1) -> (d0 - 10 + d1 - 10)> (%i, %j)
            func.call @black_box(%out) : (index) -> ()
        }
        func.return
    }
}

// CHECK_CAN:   #map = affine_map<()[s0, s1] -> (s0 + s1 - 20)>
// CHECK_CAN:   #set = affine_set<()[s0, s1, s2] : (s1 - 10 >= 0, -s1 + s0 - 9 >= 0, s2 - 10 >= 0, -s2 + s0 - 9 >= 0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index)
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index) {
// CHECK_CAN:           affine.if #set()[%arg0, %arg1, %arg2] {
// CHECK_CAN:               %0 = affine.apply #map()[%arg1, %arg2]
// CHECK_CAN:               func.call @black_box(%0) : (index) -> ()
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0 + d1 - 20)>
// CHECK_GEN:   #set = affine_set<(d0, d1)[s0] : (d0 - 10 >= 0, -d0 + s0 - 9 >= 0, d1 - 10 >= 0, -d1 + s0 - 9 >= 0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index):
// CHECK_GEN:           "affine.if"(%arg1, %arg2, %arg0) ({
// CHECK_GEN:               %0 = "affine.apply"(%arg1, %arg2) <{map = #map}> : (index, index) -> index
// CHECK_GEN:               "func.call"(%0) <{callee = @black_box}> : (index) -> ()
// CHECK_GEN:               "affine.yield"() : () -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:           }) {condition = #set} : (index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
