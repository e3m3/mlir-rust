// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<2048xi8>, %i: index) -> memref<64x4xf32>
    {
        %mr1 = memref.view %mr0[%i][] : memref<2048xi8> to memref<64x4xf32>
        func.return %mr1 : memref<64x4xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<2048xi8>, %arg1: index) -> memref<64x4xf32> {
// CHECK_CAN:           %view = memref.view %arg0[%arg1][] : memref<2048xi8> to memref<64x4xf32>
// CHECK_CAN:           return %view : memref<64x4xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<2048xi8>, index) -> memref<64x4xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<2048xi8>, %arg1: index):
// CHECK_GEN:           %0 = "memref.view"(%arg0, %arg1) : (memref<2048xi8>, index) -> memref<64x4xf32>
// CHECK_GEN:           "func.return"(%0) : (memref<64x4xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
