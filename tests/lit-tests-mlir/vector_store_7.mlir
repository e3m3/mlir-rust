// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(%value: vector<8xf32>, %base: memref<7xf32>, %i: index) -> ()
    {
        vector.store %value, %base[%i] : memref<7xf32>, vector<8xf32>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<8xf32>, %arg1: memref<7xf32>, %arg2: index) {
// CHECK_CAN:           vector.store %arg0, %arg1[%arg2] : memref<7xf32>, vector<8xf32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<8xf32>, memref<7xf32>, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<8xf32>, %arg1: memref<7xf32>, %arg2: index):
// CHECK_GEN:           "vector.store"(%arg0, %arg1, %arg2) : (vector<8xf32>, memref<7xf32>, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
