// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<?x?xf32>) -> index
    {
        %out = memref.rank %mr0 : memref<?x?xf32>
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?xf32>) -> index {
// CHECK_CAN:           %c2 = arith.constant 2 : index
// CHECK_CAN:           return %c2 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?xf32>) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?xf32>):
// CHECK_GEN:           %0 = "memref.rank"(%arg0) : (memref<?x?xf32>) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
