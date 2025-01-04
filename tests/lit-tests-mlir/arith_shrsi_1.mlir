// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%a: i8, %b: i8) -> i8
    {
        %out = arith.shrsi %a, %b : i8
        func.return %out : i8
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i8, %arg1: i8) -> i8 {
// CHECK_CAN:           %0 = arith.shrsi %arg0, %arg1 : i8
// CHECK_CAN:           return %0 : i8
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i8, i8) -> i8, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i8, %arg1: i8):
// CHECK_GEN:           %0 = "arith.shrsi"(%arg0, %arg1) : (i8, i8) -> i8
// CHECK_GEN:           "func.return"(%0) : (i8) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
