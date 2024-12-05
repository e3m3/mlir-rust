// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%v: vector<16xf32>, %i: i32) -> f32
    {
        %v0 = vector.extractelement %v[%i : i32] : vector<16xf32>
        func.return %v0 : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<16xf32>, %arg1: i32) -> f32 {
// CHECK_CAN:           %0 = vector.extractelement %arg0[%arg1 : i32] : vector<16xf32>
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<16xf32>, i32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<16xf32>, %arg1: i32):
// CHECK_GEN:           %0 = "vector.extractelement"(%arg0, %arg1) : (vector<16xf32>, i32) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
