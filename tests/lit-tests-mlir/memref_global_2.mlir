// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    memref.global "private" @x : memref<2xf32> = dense<[0.0,2.0]> {alignment = 8}
    func.func @test() -> ()
    {
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       memref.global "private" @x : memref<2xf32> = dense<[0.000000e+00, 2.000000e+00]> {alignment = 8 : i64}
// CHECK_CAN:       func.func @test() {
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "memref.global"() <{alignment = 8 : i64, initial_value = dense<[0.000000e+00, 2.000000e+00]> : tensor<2xf32>, sym_name = "x", sym_visibility = "private", type = memref<2xf32>}> : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
