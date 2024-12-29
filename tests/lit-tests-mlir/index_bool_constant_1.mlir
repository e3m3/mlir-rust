// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test() -> i1
    {
        %out = index.bool.constant true
        func.return %out : i1
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> i1 {
// CHECK_CAN:           %true = index.bool.constant true
// CHECK_CAN:           return %true : i1
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> i1, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "index.bool.constant"() <{value = true}> : () -> i1
// CHECK_GEN:           "func.return"(%0) : (i1) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
