// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  cf

module {
    func.func @test(%cond: i1, %value: f32) -> f32
    {
        cf.assert %cond, "Expected condition to be true"
        cf.br ^bb1(%value: f32)
    ^bb1(%out: f32):
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: f32) -> f32 {
// CHECK_CAN:           cf.assert %arg0, "Expected condition to be true"
// CHECK_CAN:           return %arg1 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i1, f32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: f32):
// CHECK_GEN:           "cf.assert"(%arg0) <{msg = "Expected condition to be true"}> : (i1) -> ()
// CHECK_GEN:           "cf.br"(%arg1)[^bb1] : (f32) -> ()
// CHECK_GEN:       ^bb1(%0: f32):  // pred: ^bb0
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
