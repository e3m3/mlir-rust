// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<*xf32>) -> memref<4x?xf32>
    {
        %mr1 = memref.cast %mr0 : memref<*xf32> to memref<4x?xf32>
        func.return %mr1 : memref<4x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<*xf32>) -> memref<4x?xf32> {
// CHECK_CAN:           %cast = memref.cast %arg0 : memref<*xf32> to memref<4x?xf32>
// CHECK_CAN:           return %cast : memref<4x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<*xf32>) -> memref<4x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<*xf32>):
// CHECK_GEN:           %0 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<4x?xf32>
// CHECK_GEN:           "func.return"(%0) : (memref<4x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
