// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  linalg
// CHECK-SAME:  memref

module {
    func.func @test(%a: memref<8x16xi64>, %b: memref<8x16xi64>) -> memref<8x16xi64>
    {
        %out = memref.alloc() : memref<8x16xi64>
        linalg.sub ins(%a, %b: memref<8x16xi64>, memref<8x16xi64>) outs(%out: memref<8x16xi64>)
        func.return %out : memref<8x16xi64>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<8x16xi64>, %arg1: memref<8x16xi64>) -> memref<8x16xi64> {
// CHECK_CAN:           %alloc = memref.alloc() : memref<8x16xi64>
// CHECK_CAN:           linalg.sub ins(%arg0, %arg1 : memref<8x16xi64>, memref<8x16xi64>) outs(%alloc : memref<8x16xi64>)
// CHECK_CAN:           return %alloc : memref<8x16xi64>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<8x16xi64>, memref<8x16xi64>) -> memref<8x16xi64>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<8x16xi64>, %arg1: memref<8x16xi64>):
// CHECK_GEN:           %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x16xi64>
// CHECK_GEN:           "linalg.sub"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK_GEN:           ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
// CHECK_GEN:               %1 = "arith.subi"(%arg2, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK_GEN:               "linalg.yield"(%1) : (i64) -> ()
// CHECK_GEN:           }) : (memref<8x16xi64>, memref<8x16xi64>, memref<8x16xi64>) -> ()
// CHECK_GEN:           "func.return"(%0) : (memref<8x16xi64>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
