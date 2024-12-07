// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%i: index) -> memref<8x?xf32, 1>
    {
        %mr0 = memref.alloc(%i) : memref<8x?xf32, 1>
        func.return %mr0 : memref<8x?xf32, 1>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index) -> memref<8x?xf32, 1> {
// CHECK_CAN:           %alloc = memref.alloc(%arg0) : memref<8x?xf32, 1>
// CHECK_CAN:           return %alloc : memref<8x?xf32, 1>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> memref<8x?xf32, 1>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index):
// CHECK_GEN:           %0 = "memref.alloc"(%arg0) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<8x?xf32, 1>
// CHECK_GEN:           "func.return"(%0) : (memref<8x?xf32, 1>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
