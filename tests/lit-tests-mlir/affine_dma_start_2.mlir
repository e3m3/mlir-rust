// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(
        %N: index,
        %src: memref<40x128xf32, 0>,
        %dst: memref<2x1024xf32, 1>,
        %tagbuf: memref<1xi32, 2>,
        %i: index,
        %j: index,
        %k: index,
        %l: index,
        %tag: index,
        %stride: index,
        %N_stride: index
    ) -> ()
    {
        affine.dma_start %src[%i + 3, %j], %dst[%k + 7, %l], %tagbuf[%tag], %N, %stride, %N_stride :
            memref<40x128xf32, 0>, memref<2x1024xf32, 1>, memref<1xi32, 2>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: memref<40x128xf32>, %arg2: memref<2x1024xf32, 1>, %arg3: memref<1xi32, 2>, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index) {
// CHECK_CAN:           affine.dma_start %arg1[%arg4 + 3, %arg5], %arg2[%arg6 + 7, %arg7], %arg3[%arg8], %arg0, %arg9, %arg10 : memref<40x128xf32>, memref<2x1024xf32, 1>, memref<1xi32, 2>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1) -> (d0 + 7, d1)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1) -> (d0 + 3, d1)>
// CHECK_GEN:   #map2 = affine_map<(d0) -> (d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, memref<40x128xf32>, memref<2x1024xf32, 1>, memref<1xi32, 2>, index, index, index, index, index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: memref<40x128xf32>, %arg2: memref<2x1024xf32, 1>, %arg3: memref<1xi32, 2>, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index):
// CHECK_GEN:           "affine.dma_start"(%arg1, %arg4, %arg5, %arg2, %arg6, %arg7, %arg3, %arg8, %arg0, %arg9, %arg10) {dst_map = #map, src_map = #map1, tag_map = #map2} : (memref<40x128xf32>, index, index, memref<2x1024xf32, 1>, index, index, memref<1xi32, 2>, index, index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
