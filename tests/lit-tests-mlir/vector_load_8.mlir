// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(%base: memref<100x100xf32>, %i: index, %j: index) -> vector<f32>
    {
        %v0 = vector.load %base[%i, %j] {nontemporal = true} : memref<100x100xf32>, vector<f32>
        func.return %v0 : vector<f32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index) -> vector<f32> {
// CHECK_CAN:           %0 = vector.load %arg0[%arg1, %arg2] {nontemporal = true} : memref<100x100xf32>, vector<f32>
// CHECK_CAN:           return %0 : vector<f32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<100x100xf32>, index, index) -> vector<f32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "vector.load"(%arg0, %arg1, %arg2) <{nontemporal = true}> : (memref<100x100xf32>, index, index) -> vector<f32>
// CHECK_GEN:           "func.return"(%0) : (vector<f32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
