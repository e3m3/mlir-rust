// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func

module {
    func.func private @test(i32, f64, f64) -> f64 attributes {fn_class = "kernel"}
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @test(i32, f64, f64) -> f64 attributes {fn_class = "kernel"}
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32, f64, f64) -> f64, sym_name = "test", sym_visibility = "private"}> ({
// CHECK_GEN:       }) {fn_class = "kernel"} : () -> ()
// CHECK_GEN:   }) : () -> ()
