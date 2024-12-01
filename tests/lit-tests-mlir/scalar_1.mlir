// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func private @print__Fn_Ruint_Pf64(f64) -> ()

    func.func @test() -> ()
    {
        %a = arith.constant 2.0 : f64
        %print = func.constant @print__Fn_Ruint_Pf64 : (f64) -> ()
        func.call_indirect %print(%a) : (f64) -> ()
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @print__Fn_Ruint_Pf64(f64)
// CHECK_CAN:       func.func @test() {
// CHECK_CAN:           %cst = arith.constant 2.000000e+00 : f64
// CHECK_CAN:           call @print__Fn_Ruint_Pf64(%cst) : (f64) -> ()
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f64) -> (), sym_name = "print__Fn_Ruint_Pf64", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
// CHECK_GEN:           %0 = "arith.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
// CHECK_GEN:           %1 = "func.constant"() <{value = @print__Fn_Ruint_Pf64}> : () -> ((f64) -> ())
// CHECK_GEN:           "func.call_indirect"(%1, %0) : ((f64) -> (), f64) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
