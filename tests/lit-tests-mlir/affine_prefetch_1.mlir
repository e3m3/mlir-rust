// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test(%buf: memref<400x400xi32>, %i: index, %j: index) -> ()
    {
        affine.prefetch %buf[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<400x400xi32>, %arg1: index, %arg2: index) {
// CHECK_CAN:           affine.prefetch %arg0[symbol(%arg1), symbol(%arg2) + 5], read, locality<3>, data : memref<400x400xi32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0, d1 + 5)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<400x400xi32>, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<400x400xi32>, %arg1: index, %arg2: index):
// CHECK_GEN:           "affine.prefetch"(%arg0, %arg1, %arg2) <{isDataCache = true, isWrite = false, localityHint = 3 : i32, map = #map}> : (memref<400x400xi32>, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
