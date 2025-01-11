// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%N: index, %tagbuf: memref<1xi32, 2>, %tag: index) -> ()
    {
        affine.dma_wait %tagbuf[%tag], %N : memref<1xi32, 2>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: memref<1xi32, 2>, %arg2: index) {
// CHECK_CAN:           affine.dma_wait %arg1[%arg2], %arg0 : memref<1xi32, 2>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0) -> (d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, memref<1xi32, 2>, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: memref<1xi32, 2>, %arg2: index):
// CHECK_GEN:           "affine.dma_wait"(%arg1, %arg2, %arg0) {tag_map = #map} : (memref<1xi32, 2>, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
