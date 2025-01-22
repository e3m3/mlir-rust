// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  cf

module {
    func.func @test(%cond: i1) -> ()
    {
        cf.assert %cond, "Expected condition to be true"
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i1) {
// CHECK_CAN:           cf.assert %arg0, "Expected condition to be true"
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i1) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1):
// CHECK_GEN:           "cf.assert"(%arg0) <{msg = "Expected condition to be true"}> : (i1) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
