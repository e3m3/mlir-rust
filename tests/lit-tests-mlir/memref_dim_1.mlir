// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<4x?xf32>, %i: index) -> index
    {
        %out = memref.dim %mr0, %i : memref<4x?xf32>
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<4x?xf32>, %arg1: index) -> index {
// CHECK_CAN:           %dim = memref.dim %arg0, %arg1 : memref<4x?xf32>
// CHECK_CAN:           return %dim : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<4x?xf32>, index) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<4x?xf32>, %arg1: index):
// CHECK_GEN:           %0 = "memref.dim"(%arg0, %arg1) : (memref<4x?xf32>, index) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
