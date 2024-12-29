// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%value: f32, %base: memref<100x100xf32>, %i: index, %j: index) -> ()
    {
        affine.store %value, %base[%i + 3, %j + 7] : memref<100x100xf32>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: f32, %arg1: memref<100x100xf32>, %arg2: index, %arg3: index) {
// CHECK_CAN:           affine.store %arg0, %arg1[symbol(%arg2) + 3, symbol(%arg3) + 7] : memref<100x100xf32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0 + 3, d1 + 7)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32, memref<100x100xf32>, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f32, %arg1: memref<100x100xf32>, %arg2: index, %arg3: index):
// CHECK_GEN:           "affine.store"(%arg0, %arg1, %arg2, %arg3) <{map = #map}> : (f32, memref<100x100xf32>, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
