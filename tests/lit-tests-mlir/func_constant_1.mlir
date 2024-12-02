// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func

module {
    func.func private @callee(f64, f64) -> f64

    func.func @test() -> ((f64, f64) -> f64)
    {
        %out = func.constant @callee : (f64, f64) -> f64
        func.return %out : (f64, f64) -> f64
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @callee(f64, f64) -> f64
// CHECK_CAN:       func.func @test() -> ((f64, f64) -> f64) {
// CHECK_CAN:           %f = constant @callee : (f64, f64) -> f64
// CHECK_CAN:           return %f : (f64, f64) -> f64
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f64, f64) -> f64, sym_name = "callee", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> ((f64, f64) -> f64), sym_name = "test"}> ({
// CHECK_GEN:           %0 = "func.constant"() <{value = @callee}> : () -> ((f64, f64) -> f64)
// CHECK_GEN:           "func.return"(%0) : ((f64, f64) -> f64) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
