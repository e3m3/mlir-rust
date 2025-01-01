// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%in: i32) -> f32
    {
        %out = arith.bitcast %in : i32 to f32
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i32) -> f32 {
// CHECK_CAN:           %0 = arith.bitcast %arg0 : i32 to f32
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i32):
// CHECK_GEN:           %0 = "arith.bitcast"(%arg0) : (i32) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
