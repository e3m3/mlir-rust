// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box(i32) -> (i32, i32)
    func.func @test(%cond: i1, %a: i32, %b: i32) -> (i32, i32)
    {
        %out0, %out1 = scf.if %cond -> (i32, i32) {
            %val:2 = func.call @black_box(%a) : (i32) -> (i32, i32)
            scf.yield %val#0, %val#1 : i32, i32
        } else {
            %val:2 = func.call @black_box(%b) : (i32) -> (i32, i32)
            scf.yield %val#0, %val#1 : i32, i32
        }
        func.return %out0, %out1: i32, i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(i32) -> (i32, i32)
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32, i32) {
// CHECK_CAN:           %0:2 = scf.if %arg0 -> (i32, i32) {
// CHECK_CAN:               %1:2 = func.call @black_box(%arg1) : (i32) -> (i32, i32)
// CHECK_CAN:               scf.yield %1#0, %1#1 : i32, i32
// CHECK_CAN:           } else {
// CHECK_CAN:               %1:2 = func.call @black_box(%arg2) : (i32) -> (i32, i32)
// CHECK_CAN:               scf.yield %1#0, %1#1 : i32, i32
// CHECK_CAN:           }
// CHECK_CAN:           return %0#0, %0#1 : i32, i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32) -> (i32, i32), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i1, i32, i32) -> (i32, i32), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
// CHECK_GEN:           %0:2 = "scf.if"(%arg0) ({
// CHECK_GEN:               %2:2 = "func.call"(%arg1) <{callee = @black_box}> : (i32) -> (i32, i32)
// CHECK_GEN:               "scf.yield"(%2#0, %2#1) : (i32, i32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               %1:2 = "func.call"(%arg2) <{callee = @black_box}> : (i32) -> (i32, i32)
// CHECK_GEN:               "scf.yield"(%1#0, %1#1) : (i32, i32) -> ()
// CHECK_GEN:           }) : (i1) -> (i32, i32)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (i32, i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
