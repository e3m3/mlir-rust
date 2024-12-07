// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<8x?xi32, strided<[?, ?], offset: ?>>, %i: index, %j: index) -> i32
    {
        %out = memref.load %mr0[%i, %j] : memref<8x?xi32, strided<[?, ?], offset: ?>>
        func.return %out : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<8x?xi32, strided<[?, ?], offset: ?>>, %arg1: index, %arg2: index) -> i32 {
// CHECK_CAN:           %0 = memref.load %arg0[%arg1, %arg2] : memref<8x?xi32, strided<[?, ?], offset: ?>>
// CHECK_CAN:           return %0 : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<8x?xi32, strided<[?, ?], offset: ?>>, index, index) -> i32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<8x?xi32, strided<[?, ?], offset: ?>>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "memref.load"(%arg0, %arg1, %arg2) : (memref<8x?xi32, strided<[?, ?], offset: ?>>, index, index) -> i32
// CHECK_GEN:           "func.return"(%0) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
