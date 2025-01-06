// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%a: f32, %b: f32) -> i1
    {
        %out = arith.cmpf oeq, %a, %b : f32
        func.return %out : i1
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: f32, %arg1: f32) -> i1 {
// CHECK_CAN:           %0 = arith.cmpf oeq, %arg0, %arg1 : f32
// CHECK_CAN:           return %0 : i1
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32) -> i1, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f32, %arg1: f32):
// CHECK_GEN:           %0 = "arith.cmpf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (f32, f32) -> i1
// CHECK_GEN:           "func.return"(%0) : (i1) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
