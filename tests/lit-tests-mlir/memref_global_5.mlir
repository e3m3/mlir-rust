// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    memref.global constant @c : memref<1xi32> = dense<1>
    func.func @test() -> ()
    {
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       memref.global constant @c : memref<1xi32> = dense<1>
// CHECK_CAN:       func.func @test() {
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "memref.global"() <{constant, initial_value = dense<1> : tensor<1xi32>, sym_name = "c", type = memref<1xi32>}> : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
