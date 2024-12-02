// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func

module {
    func.func private @callee(f64, f64) -> f64

    func.func @test(%a: f64, %b: f64) -> f64
    {
        %callee = func.constant @callee : (f64, f64) -> f64
        %out = func.call_indirect %callee(%a, %b) : (f64, f64) -> f64
        func.return %out : f64
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @callee(f64, f64) -> f64
// CHECK_CAN:       func.func @test(%arg0: f64, %arg1: f64) -> f64 {
// CHECK_CAN:           %0 = call @callee(%arg0, %arg1) : (f64, f64) -> f64
// CHECK_CAN:           return %0 : f64
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f64, f64) -> f64, sym_name = "callee", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f64, f64) -> f64, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK_GEN:           %0 = "func.constant"() <{value = @callee}> : () -> ((f64, f64) -> f64)
// CHECK_GEN:           %1 = "func.call_indirect"(%0, %arg0, %arg1) : ((f64, f64) -> f64, f64, f64) -> f64
// CHECK_GEN:           "func.return"(%1) : (f64) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
