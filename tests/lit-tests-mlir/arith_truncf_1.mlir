// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%in: f64) -> f16
    {
        %out = arith.truncf %in {
           fastmath = #arith.fastmath<fast>,
           roundingmode = 1 : i32
        } : f64 to f16
        func.return %out : f16
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: f64) -> f16 {
// CHECK_CAN:           %0 = arith.truncf %arg0 downward fastmath<fast> : f64 to f16
// CHECK_CAN:           return %0 : f16
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f64) -> f16, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f64):
// CHECK_GEN:           %0 = "arith.truncf"(%arg0) <{fastmath = #arith.fastmath<fast>, roundingmode = 1 : i32}> : (f64) -> f16
// CHECK_GEN:           "func.return"(%0) : (f16) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
