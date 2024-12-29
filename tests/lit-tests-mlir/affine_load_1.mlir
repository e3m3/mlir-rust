// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%base: memref<100x100xf32>, %i: index, %j: index) -> f32
    {
        %out = affine.load %base[%i + 3, %j + 7] : memref<100x100xf32>
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK_CAN:           %0 = affine.load %arg0[symbol(%arg1) + 3, symbol(%arg2) + 7] : memref<100x100xf32>
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0 + 3, d1 + 7)>
// CHECK_GEN:       "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<100x100xf32>, index, index) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "affine.load"(%arg0, %arg1, %arg2) <{map = #map}> : (memref<100x100xf32>, index, index) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
