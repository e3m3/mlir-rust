// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(
        %mr0: memref<12x4xf32, strided<[4, 1], offset: 5>>
    ) -> memref<12x4xf32, strided<[?, ?], offset: ?>> {
        %mr1 = memref.cast %mr0 : memref<12x4xf32, strided<[4, 1], offset: 5>>
            to memref<12x4xf32, strided<[?, ?], offset: ?>>
        func.return %mr1 : memref<12x4xf32, strided<[?, ?], offset: ?>>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<12x4xf32, strided<[4, 1], offset: 5>>) -> memref<12x4xf32, strided<[?, ?], offset: ?>> {
// CHECK_CAN:           %cast = memref.cast %arg0 : memref<12x4xf32, strided<[4, 1], offset: 5>> to memref<12x4xf32, strided<[?, ?], offset: ?>>
// CHECK_CAN:           return %cast : memref<12x4xf32, strided<[?, ?], offset: ?>>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<12x4xf32, strided<[4, 1], offset: 5>>) -> memref<12x4xf32, strided<[?, ?], offset: ?>>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<12x4xf32, strided<[4, 1], offset: 5>>):
// CHECK_GEN:           %0 = "memref.cast"(%arg0) : (memref<12x4xf32, strided<[4, 1], offset: 5>>) -> memref<12x4xf32, strided<[?, ?], offset: ?>>
// CHECK_GEN:           "func.return"(%0) : (memref<12x4xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
