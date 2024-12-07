// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<2048xi8>, %i: index, %x: index, %y: index) -> memref<?x4x?xf32>
    {
        %mr1 = memref.view %mr0[%i][%x, %y] : memref<2048xi8> to memref<?x4x?xf32>
        func.return %mr1 : memref<?x4x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<2048xi8>, %arg1: index, %arg2: index, %arg3: index) -> memref<?x4x?xf32> {
// CHECK_CAN:           %view = memref.view %arg0[%arg1][%arg2, %arg3] : memref<2048xi8> to memref<?x4x?xf32>
// CHECK_CAN:           return %view : memref<?x4x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<2048xi8>, index, index, index) -> memref<?x4x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<2048xi8>, %arg1: index, %arg2: index, %arg3: index):
// CHECK_GEN:           %0 = "memref.view"(%arg0, %arg1, %arg2, %arg3) : (memref<2048xi8>, index, index, index) -> memref<?x4x?xf32>
// CHECK_GEN:           "func.return"(%0) : (memref<?x4x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
