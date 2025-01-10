// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func @test(%i: index, %j: index) -> index
    {
        %out = affine.max affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%i)[%j]
        func.return %out : index
    }
}

// CHECK_CAN:   #map = affine_map<()[s0, s1] -> (1000, s1 + 512, s0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> index {
// CHECK_CAN:           %0 = affine.max #map()[%arg1, %arg0]
// CHECK_CAN:           return %0 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0)[s0] -> (1000, d0 + 512, s0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "affine.max"(%arg0, %arg1) <{map = #map}> : (index, index) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
