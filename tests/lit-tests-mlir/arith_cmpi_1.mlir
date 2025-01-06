// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%a: i32, %b: i32) -> i1
    {
        %out = arith.cmpi slt, %a, %b : i32
        func.return %out : i1
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i32, %arg1: i32) -> i1 {
// CHECK_CAN:           %0 = arith.cmpi slt, %arg0, %arg1 : i32
// CHECK_CAN:           return %0 : i1
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32, i32) -> i1, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i32, %arg1: i32):
// CHECK_GEN:           %0 = "arith.cmpi"(%arg0, %arg1) <{predicate = 2 : i64}> : (i32, i32) -> i1
// CHECK_GEN:           "func.return"(%0) : (i1) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
