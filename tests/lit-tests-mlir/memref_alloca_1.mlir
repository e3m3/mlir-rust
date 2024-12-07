// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test() -> memref<8x64xf32>
    {
        %mr0 = memref.alloca() : memref<8x64xf32>
        func.return %mr0 : memref<8x64xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> memref<8x64xf32> {
// CHECK_CAN:           %alloca = memref.alloca() : memref<8x64xf32>
// CHECK_CAN:           return %alloca : memref<8x64xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> memref<8x64xf32>, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xf32>
// CHECK_GEN:           "func.return"(%0) : (memref<8x64xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
