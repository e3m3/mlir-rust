// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  cf

module {
    func.func private @black_box_0(i32) -> index
    func.func private @black_box_1(f32) -> index
    func.func @test(%cond: i1, %a: i32, %b: f32) -> index
    {
        cf.cond_br %cond, ^bb1(%a: i32), ^bb2(%b: f32)
    ^bb1(%in_0: i32):
        %res_0 = func.call @black_box_0(%in_0) : (i32) -> index
        cf.br ^bb3(%res_0: index)
    ^bb2(%in_1: f32):
        %res_1 = func.call @black_box_1(%in_1) : (f32) -> index
        cf.br ^bb3(%res_1: index)
    ^bb3(%out: index):
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(i32) -> index
// CHECK_CAN:       func.func private @black_box_1(f32) -> index
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: i32, %arg2: f32) -> index {
// CHECK_CAN:           cf.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : f32)
// CHECK_CAN:       ^bb1(%0: i32):  // pred: ^bb0
// CHECK_CAN:           %1 = call @black_box_0(%0) : (i32) -> index
// CHECK_CAN:           cf.br ^bb3(%1 : index)
// CHECK_CAN:       ^bb2(%2: f32):  // pred: ^bb0
// CHECK_CAN:           %3 = call @black_box_1(%2) : (f32) -> index
// CHECK_CAN:           cf.br ^bb3(%3 : index)
// CHECK_CAN:       ^bb3(%4: index):  // 2 preds: ^bb1, ^bb2
// CHECK_CAN:           return %4 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32) -> index, sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32) -> index, sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i1, i32, f32) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: i32, %arg2: f32):
// CHECK_GEN:           "cf.cond_br"(%arg0, %arg1, %arg2)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, f32) -> ()
// CHECK_GEN:       ^bb1(%0: i32):  // pred: ^bb0
// CHECK_GEN:           %1 = "func.call"(%0) <{callee = @black_box_0}> : (i32) -> index
// CHECK_GEN:           "cf.br"(%1)[^bb3] : (index) -> ()
// CHECK_GEN:       ^bb2(%2: f32):  // pred: ^bb0
// CHECK_GEN:           %3 = "func.call"(%2) <{callee = @black_box_1}> : (f32) -> index
// CHECK_GEN:           "cf.br"(%3)[^bb3] : (index) -> ()
// CHECK_GEN:       ^bb3(%4: index):  // 2 preds: ^bb1, ^bb2
// CHECK_GEN:           "func.return"(%4) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
