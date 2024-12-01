// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN
// COM: [1]: https://mlir.llvm.org/docs/Dialects/UBOps/#ubpoison-ubpoisonop

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  ub

module {
    func.func @test() -> i32
    {
        %0 = ub.poison : i32
        return %0 : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> i32 {
// CHECK_CAN:           %0 = ub.poison : i32
// CHECK_CAN:           return %0 : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> i32, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "ub.poison"() <{value = #ub.poison}> : () -> i32
// CHECK_GEN:           "func.return"(%0) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
