// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func private @print__Fn_Ruint_Pf64(f64) -> ()

    func.func @test(%input: f64) -> ()
    {
        %0 = arith.constant 2.0 : f64
        %a = arith.addf %0, %input : f64
        %print = func.constant @print__Fn_Ruint_Pf64 : (f64) -> ()
        func.call_indirect %print(%a) : (f64) -> ()
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @print__Fn_Ruint_Pf64(f64)
// CHECK_CAN:       func.func @test(%arg0: f64) {
// CHECK_CAN:           %cst = arith.constant 2.000000e+00 : f64
// CHECK_CAN:           %0 = arith.addf %arg0, %cst : f64
// CHECK_CAN:           call @print__Fn_Ruint_Pf64(%0) : (f64) -> ()
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f64) -> (), sym_name = "print__Fn_Ruint_Pf64", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f64) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f64):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
// CHECK_GEN:           %1 = "arith.addf"(%0, %arg0) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK_GEN:           %2 = "func.constant"() <{value = @print__Fn_Ruint_Pf64}> : () -> ((f64) -> ())
// CHECK_GEN:           "func.call_indirect"(%2, %1) : ((f64) -> (), f64) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
