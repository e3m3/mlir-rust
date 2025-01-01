// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%in: i64) -> i16
    {
        %out = arith.trunci %in : i64 to i16
        func.return %out : i16
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i64) -> i16 {
// CHECK_CAN:           %0 = arith.trunci %arg0 : i64 to i16
// CHECK_CAN:           return %0 : i16
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i64) -> i16, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i64):
// CHECK_GEN:           %0 = "arith.trunci"(%arg0) : (i64) -> i16
// CHECK_GEN:           "func.return"(%0) : (i16) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
