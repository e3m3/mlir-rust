// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  memref

module {
    func.func @test(%a: memref<8x16xf32>) -> memref<8x16xf32>
    {
        %out = memref.alloc() : memref<8x16xf32>
        linalg.abs ins(%a: memref<8x16xf32>) outs(%out: memref<8x16xf32>)
        func.return %out : memref<8x16xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<8x16xf32>) -> memref<8x16xf32> {
// CHECK_CAN:           %alloc = memref.alloc() : memref<8x16xf32>
// CHECK_CAN:           linalg.abs ins(%arg0 : memref<8x16xf32>) outs(%alloc : memref<8x16xf32>)
// CHECK_CAN:           return %alloc : memref<8x16xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<8x16xf32>) -> memref<8x16xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<8x16xf32>):
// CHECK_GEN:           %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x16xf32>
// CHECK_GEN:           "linalg.abs"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK_GEN:           ^bb0(%arg1: f32, %arg2: f32):
// CHECK_GEN:               %1 = "math.absf"(%arg1) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK_GEN:               "linalg.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }) : (memref<8x16xf32>, memref<8x16xf32>) -> ()
// CHECK_GEN:           "func.return"(%0) : (memref<8x16xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
