// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box_0(f32) -> (i1)
    func.func private @black_box_1(f32) -> (f32)
    func.func @test(%init: f32) -> f32
    {
        %out = scf.while (%acc = %init) : (f32) -> (f32) {
            %cond = func.call @black_box_0(%acc) : (f32) -> (i1)
            scf.condition(%cond) %acc : f32
        } do {
        ^bb0(%acc: f32):
            %next = func.call @black_box_1(%acc) : (f32) -> (f32)
            scf.yield %next : f32
        }
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(f32) -> i1
// CHECK_CAN:       func.func private @black_box_1(f32) -> f32
// CHECK_CAN:       func.func @test(%arg0: f32) -> f32 {
// CHECK_CAN:       %0 = scf.while (%arg1 = %arg0) : (f32) -> f32 {
// CHECK_CAN:           %1 = func.call @black_box_0(%arg1) : (f32) -> i1
// CHECK_CAN:               scf.condition(%1) %arg1 : f32
// CHECK_CAN:           } do {
// CHECK_CAN:           ^bb0(%arg1: f32):
// CHECK_CAN:               %1 = func.call @black_box_1(%arg1) : (f32) -> f32
// CHECK_CAN:               scf.yield %1 : f32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> i1, sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> f32, sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: f32):
// CHECK_GEN:           %0 = "scf.while"(%arg0) ({
// CHECK_GEN:           ^bb0(%arg2: f32):
// CHECK_GEN:               %2 = "func.call"(%arg2) <{callee = @black_box_0}> : (f32) -> i1
// CHECK_GEN:               "scf.condition"(%2, %arg2) : (i1, f32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:           ^bb0(%arg1: f32):
// CHECK_GEN:               %1 = "func.call"(%arg1) <{callee = @black_box_1}> : (f32) -> f32
// CHECK_GEN:               "scf.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }) : (f32) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
