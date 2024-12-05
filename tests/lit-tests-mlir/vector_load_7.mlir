// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(%base: memref<7xf32>) -> vector<8xf32>
    {
        %i = arith.constant 0 : index
        %v0 = vector.load %base[%i] : memref<7xf32>, vector<8xf32>
        func.return %v0 : vector<8xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<7xf32>) -> vector<8xf32> {
// CHECK_CAN:           %c0 = arith.constant 0 : index
// CHECK_CAN:           %0 = vector.load %arg0[%c0] : memref<7xf32>, vector<8xf32>
// CHECK_CAN:           return %0 : vector<8xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<7xf32>) -> vector<8xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<7xf32>):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK_GEN:           %1 = "vector.load"(%arg0, %0) : (memref<7xf32>, index) -> vector<8xf32>
// CHECK_GEN:           "func.return"(%1) : (vector<8xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
