// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  memref

module {
    func.func @test(%a: memref<16x64xf32>) -> memref<64x16xf32>
    {
        %out = memref.alloc() : memref<64x16xf32>
        linalg.transpose ins(%a: memref<16x64xf32>) outs(%out: memref<64x16xf32>) permutation = [1, 0]
        func.return %out : memref<64x16xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<16x64xf32>) -> memref<64x16xf32> {
// CHECK_CAN:           %alloc = memref.alloc() : memref<64x16xf32>
// CHECK_CAN:           linalg.transpose ins(%arg0 : memref<16x64xf32>) outs(%alloc : memref<64x16xf32>) permutation = [1, 0]
// CHECK_CAN:           return %alloc : memref<64x16xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<16x64xf32>) -> memref<64x16xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<16x64xf32>):
// CHECK_GEN:           %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x16xf32>
// CHECK_GEN:           "linalg.transpose"(%arg0, %0) <{permutation = array<i64: 1, 0>}> ({
// CHECK_GEN:           ^bb0(%arg1: f32, %arg2: f32):
// CHECK_GEN:               "linalg.yield"(%arg1) : (f32) -> ()
// CHECK_GEN:           }) : (memref<16x64xf32>, memref<64x16xf32>) -> ()
// CHECK_GEN:           "func.return"(%0) : (memref<64x16xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
