// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  cf

module {
    func.func private @black_box_0(f32, f32) -> ()
    func.func private @black_box_1(i16, i16) -> ()
    func.func @test(%flag: i32, %d: i1, %a0: f32, %a1: f32, %b0: i16, %b1: i16) -> ()
    {
        cf.switch %flag : i32, [
            default: ^bb1(%d: i1),
            100: ^bb2(%a0, %a1 : f32, f32),
            200: ^bb3(%b0, %b1 : i16, i16)
        ]
    ^bb1(%cond: i1):
        cf.assert %cond, "Expected true for default case"
        cf.br ^bb4
    ^bb2(%in_a0: f32, %in_a1: f32):
        func.call @black_box_0(%in_a0, %in_a1) : (f32, f32) -> ()
        cf.br ^bb4
    ^bb3(%in_b0: i16, %in_b1: i16):
        func.call @black_box_1(%in_b0, %in_b1) : (i16, i16) -> ()
        cf.br ^bb4
    ^bb4:
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(f32, f32)
// CHECK_CAN:       func.func private @black_box_1(i16, i16)
// CHECK_CAN:       func.func @test(%arg0: i32, %arg1: i1, %arg2: f32, %arg3: f32, %arg4: i16, %arg5: i16) {
// CHECK_CAN:           cf.switch %arg0 : i32, [
// CHECK_CAN:               default: ^bb1(%arg1 : i1),
// CHECK_CAN:               100: ^bb2(%arg2, %arg3 : f32, f32),
// CHECK_CAN:               200: ^bb3(%arg4, %arg5 : i16, i16)
// CHECK_CAN:           ]
// CHECK_CAN:       ^bb1(%0: i1):  // pred: ^bb0
// CHECK_CAN:           cf.assert %0, "Expected true for default case"
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb2(%1: f32, %2: f32):  // pred: ^bb0
// CHECK_CAN:           call @black_box_0(%1, %2) : (f32, f32) -> ()
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb3(%3: i16, %4: i16):  // pred: ^bb0
// CHECK_CAN:           call @black_box_1(%3, %4) : (i16, i16) -> ()
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32) -> (), sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i16, i16) -> (), sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i32, i1, f32, f32, i16, i16) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i32, %arg1: i1, %arg2: f32, %arg3: f32, %arg4: i16, %arg5: i16):
// CHECK_GEN:           "cf.switch"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)[^bb1, ^bb2, ^bb3] <{case_operand_segments = array<i32: 2, 2>, case_values = dense<[100, 200]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 4>}> : (i32, i1, f32, f32, i16, i16) -> ()
// CHECK_GEN:       ^bb1(%0: i1):  // pred: ^bb0
// CHECK_GEN:           "cf.assert"(%0) <{msg = "Expected true for default case"}> : (i1) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb2(%1: f32, %2: f32):  // pred: ^bb0
// CHECK_GEN:           "func.call"(%1, %2) <{callee = @black_box_0}> : (f32, f32) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb3(%3: i16, %4: i16):  // pred: ^bb0
// CHECK_GEN:           "func.call"(%3, %4) <{callee = @black_box_1}> : (i16, i16) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
