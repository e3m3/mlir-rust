// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%a: f32, %b: f32, %c: f32, %d: f32) -> vector<4xf32>
    {
        %v = vector.from_elements %a, %b, %c, %d : vector<4xf32>
        func.return %v : vector<4xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> vector<4xf32> {
// CHECK_CAN:           %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<4xf32>
// CHECK_CAN:           return %0 : vector<4xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32, f32, f32) -> vector<4xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
// CHECK_GEN:           %0 = "vector.from_elements"(%arg0, %arg1, %arg2, %arg3) : (f32, f32, f32, f32) -> vector<4xf32>
// CHECK_GEN:           "func.return"(%0) : (vector<4xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
