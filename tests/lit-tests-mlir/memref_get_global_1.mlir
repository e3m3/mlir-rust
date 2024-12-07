// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    memref.global "private" @foo : memref<2xf32>
    func.func @test() -> memref<2xf32>
    {
        %mr0 = memref.get_global @foo : memref<2xf32>
        func.return %mr0 : memref<2xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       memref.global "private" @foo : memref<2xf32>
// CHECK_CAN:       func.func @test() -> memref<2xf32> {
// CHECK_CAN:           %0 = memref.get_global @foo : memref<2xf32>
// CHECK_CAN:           return %0 : memref<2xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "memref.global"() <{sym_name = "foo", sym_visibility = "private", type = memref<2xf32>}> : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> memref<2xf32>, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "memref.get_global"() <{name = @foo}> : () -> memref<2xf32>
// CHECK_GEN:           "func.return"(%0) : (memref<2xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
