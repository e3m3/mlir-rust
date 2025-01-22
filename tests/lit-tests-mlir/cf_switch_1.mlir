// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  cf

module {
    func.func private @black_box_0(f32) -> ()
    func.func private @black_box_1(f32) -> ()
    func.func private @black_box_2(f32) -> ()
    func.func @test(%flag: i32, %a: f32, %b: f32, %c: f32) -> ()
    {
        cf.switch %flag : i32, [
            default: ^bb1(%a: f32),
            100: ^bb2(%b: f32),
            200: ^bb3(%c: f32)
        ]
    ^bb1(%in_0: f32):
        func.call @black_box_0(%in_0) : (f32) -> ()
        cf.br ^bb4
    ^bb2(%in_1: f32):
        func.call @black_box_1(%in_1) : (f32) -> ()
        cf.br ^bb4
    ^bb3(%in_2: f32):
        func.call @black_box_2(%in_2) : (f32) -> ()
        cf.br ^bb4
    ^bb4:
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(f32)
// CHECK_CAN:       func.func private @black_box_1(f32)
// CHECK_CAN:       func.func private @black_box_2(f32)
// CHECK_CAN:       func.func @test(%arg0: i32, %arg1: f32, %arg2: f32, %arg3: f32) {
// CHECK_CAN:           cf.switch %arg0 : i32, [
// CHECK_CAN:               default: ^bb1(%arg1 : f32),
// CHECK_CAN:               100: ^bb2(%arg2 : f32),
// CHECK_CAN:               200: ^bb3(%arg3 : f32)
// CHECK_CAN:           ]
// CHECK_CAN:       ^bb1(%0: f32):  // pred: ^bb0
// CHECK_CAN:           call @black_box_0(%0) : (f32) -> ()
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb2(%1: f32):  // pred: ^bb0
// CHECK_CAN:           call @black_box_1(%1) : (f32) -> ()
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb3(%2: f32):  // pred: ^bb0
// CHECK_CAN:           call @black_box_2(%2) : (f32) -> ()
// CHECK_CAN:           cf.br ^bb4
// CHECK_CAN:       ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> (), sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> (), sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> (), sym_name = "black_box_2", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i32, f32, f32, f32) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i32, %arg1: f32, %arg2: f32, %arg3: f32):
// CHECK_GEN:           "cf.switch"(%arg0, %arg1, %arg2, %arg3)[^bb1, ^bb2, ^bb3] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[100, 200]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
// CHECK_GEN:       ^bb1(%0: f32):  // pred: ^bb0
// CHECK_GEN:           "func.call"(%0) <{callee = @black_box_0}> : (f32) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb2(%1: f32):  // pred: ^bb0
// CHECK_GEN:           "func.call"(%1) <{callee = @black_box_1}> : (f32) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb3(%2: f32):  // pred: ^bb0
// CHECK_GEN:           "func.call"(%2) <{callee = @black_box_2}> : (f32) -> ()
// CHECK_GEN:           "cf.br"()[^bb4] : () -> ()
// CHECK_GEN:       ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
