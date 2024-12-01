// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test() -> f64
    {
        %a = arith.constant 2.0 : f64
        func.return %a : f64
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> f64 {
// CHECK_CAN:           %cst = arith.constant 2.000000e+00 : f64
// CHECK_CAN:           return %cst : f64
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> f64, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "arith.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
// CHECK_GEN:           "func.return"(%0) : (f64) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
