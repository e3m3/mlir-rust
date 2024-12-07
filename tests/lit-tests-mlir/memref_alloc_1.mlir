// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test() -> memref<8x64xf32, 1>
    {
        %mr0 = memref.alloc() : memref<8x64xf32, 1>
        func.return %mr0 : memref<8x64xf32, 1>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> memref<8x64xf32, 1> {
// CHECK_CAN:           %alloc = memref.alloc() : memref<8x64xf32, 1>
// CHECK_CAN:           return %alloc : memref<8x64xf32, 1>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> memref<8x64xf32, 1>, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xf32, 1>
// CHECK_GEN:           "func.return"(%0) : (memref<8x64xf32, 1>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
