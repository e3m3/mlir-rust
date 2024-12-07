// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// COM: Unable to pass an integer literal as a store operation index like shown in [1].
// COM: [1]: https://mlir.llvm.org/docs/Dialects/MemRef/#memrefstore-memrefstoreop

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(
        %mr0: memref<4x?xf32, strided<[?, ?], offset: ?>, 0>,
        %i: index,
        %value: f32
    ) -> () {
        %j = arith.constant 1023 : index // See [1].
        memref.store %value, %mr0[%i, %j] : memref<4x?xf32, strided<[?, ?], offset: ?>, 0>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<4x?xf32, strided<[?, ?], offset: ?>>, %arg1: index, %arg2: f32) {
// CHECK_CAN:           %c1023 = arith.constant 1023 : index
// CHECK_CAN:           memref.store %arg2, %arg0[%arg1, %c1023] : memref<4x?xf32, strided<[?, ?], offset: ?>>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<4x?xf32, strided<[?, ?], offset: ?>>, index, f32) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<4x?xf32, strided<[?, ?], offset: ?>>, %arg1: index, %arg2: f32):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 1023 : index}> : () -> index
// CHECK_GEN:           "memref.store"(%arg2, %arg0, %arg1, %0) : (f32, memref<4x?xf32, strided<[?, ?], offset: ?>>, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
