// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test() -> i8
    {
        %a = arith.constant -3 : i8
        func.return %a : i8
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> i8 {
// CHECK_CAN:           %c-3_i8 = arith.constant -3 : i8
// CHECK_CAN:           return %c-3_i8 : i8
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> i8, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "arith.constant"() <{value = -3 : i8}> : () -> i8
// CHECK_GEN:           "func.return"(%0) : (i8) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
