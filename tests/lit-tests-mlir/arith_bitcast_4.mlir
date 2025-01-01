// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%in: memref<?x?xf16>) -> memref<?x?xi16>
    {
        %out = arith.bitcast %in : memref<?x?xf16> to memref<?x?xi16>
        func.return %out : memref<?x?xi16>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?xf16>) -> memref<?x?xi16> {
// CHECK_CAN:           %0 = arith.bitcast %arg0 : memref<?x?xf16> to memref<?x?xi16>
// CHECK_CAN:           return %0 : memref<?x?xi16>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?xf16>) -> memref<?x?xi16>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?xf16>):
// CHECK_GEN:           %0 = "arith.bitcast"(%arg0) : (memref<?x?xf16>) -> memref<?x?xi16>
// CHECK_GEN:           "func.return"(%0) : (memref<?x?xi16>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
