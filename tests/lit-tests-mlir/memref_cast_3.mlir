// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(
        %mr0: memref<12x4xf32, strided<[?, ?], offset: ?>>
    ) -> memref<12x4xf32, strided<[4, 1], offset: 5>> {
        %mr1 = memref.cast %mr0 : memref<12x4xf32, strided<[?, ?], offset: ?>>
            to memref<12x4xf32, strided<[4, 1], offset: 5>>
        func.return %mr1 : memref<12x4xf32, strided<[4, 1], offset: 5>>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<12x4xf32, strided<[?, ?], offset: ?>>) -> memref<12x4xf32, strided<[4, 1], offset: 5>> {
// CHECK_CAN:           %cast = memref.cast %arg0 : memref<12x4xf32, strided<[?, ?], offset: ?>> to memref<12x4xf32, strided<[4, 1], offset: 5>>
// CHECK_CAN:           return %cast : memref<12x4xf32, strided<[4, 1], offset: 5>>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<12x4xf32, strided<[?, ?], offset: ?>>) -> memref<12x4xf32, strided<[4, 1], offset: 5>>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<12x4xf32, strided<[?, ?], offset: ?>>):
// CHECK_GEN:           %0 = "memref.cast"(%arg0) : (memref<12x4xf32, strided<[?, ?], offset: ?>>) -> memref<12x4xf32, strided<[4, 1], offset: 5>>
// CHECK_GEN:           "func.return"(%0) : (memref<12x4xf32, strided<[4, 1], offset: 5>>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
