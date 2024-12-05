// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%a: f8E5M2, %b: f8E5M2) -> f8E5M2
    {
        %out = arith.subf %a, %b {fastmath = #arith.fastmath<fast>} : f8E5M2
        func.return %out : f8E5M2
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: f8E5M2, %arg1: f8E5M2) -> f8E5M2 {
// CHECK_CAN:           %0 = arith.subf %arg0, %arg1 fastmath<fast> : f8E5M2
// CHECK_CAN:           return %0 : f8E5M2
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f8E5M2, f8E5M2) -> f8E5M2, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f8E5M2, %arg1: f8E5M2):
// CHECK_GEN:           %0 = "arith.subf"(%arg0, %arg1) <{fastmath = #arith.fastmath<fast>}> : (f8E5M2, f8E5M2) -> f8E5M2
// CHECK_GEN:           "func.return"(%0) : (f8E5M2) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
