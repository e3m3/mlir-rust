// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<?xf32>, %mr1: memref<?xf32>) -> ()
    {
        memref.copy %mr0, %mr1 : memref<?xf32> to memref<?xf32>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
// CHECK_CAN:           memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?xf32>, memref<?xf32>) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?xf32>, %arg1: memref<?xf32>):
// CHECK_GEN:           "memref.copy"(%arg0, %arg1) : (memref<?xf32>, memref<?xf32>) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
