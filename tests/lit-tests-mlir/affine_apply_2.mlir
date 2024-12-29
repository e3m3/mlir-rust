// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func @test(%i: index, %s: index) -> index
    {
        %out = affine.apply affine_map<(i)[s0] -> (i + s0)> (%i)[%s]
        func.return %out : index
    }
}

// CHECK_CAN:   #map = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> index {
// CHECK_CAN:           %0 = affine.apply #map()[%arg1, %arg0]
// CHECK_CAN:           return %0 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "affine.apply"(%arg0, %arg1) <{map = #map}> : (index, index) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
