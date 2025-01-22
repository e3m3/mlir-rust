// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  cf
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box_0() -> f32
    func.func private @black_box_1() -> f32
    func.func @test(%cond_0: i1, %cond_1: i1, %default: f32) -> f32
    {
        %out = scf.if %cond_0 -> f32 {
            %res = scf.execute_region -> f32 {
            ^bb0:
                cf.cond_br %cond_1, ^bb1, ^bb2
            ^bb1:
                %value_bb1 = func.call @black_box_0() : () -> f32
                cf.br ^bb3(%value_bb1: f32)
            ^bb2:
                %value_bb2 = func.call @black_box_1() : () -> f32
                cf.br ^bb3(%value_bb2: f32)
            ^bb3(%value: f32):
                scf.yield %value : f32
            }
            scf.yield %res : f32
        } else {
            scf.yield %default : f32
        }
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0() -> f32
// CHECK_CAN:       func.func private @black_box_1() -> f32
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: i1, %arg2: f32) -> f32 {
// CHECK_CAN:           %0 = scf.if %arg0 -> (f32) {
// CHECK_CAN:               %1 = scf.execute_region -> f32 {
// CHECK_CAN:                   cf.cond_br %arg1, ^bb1, ^bb2
// CHECK_CAN:               ^bb1:  // pred: ^bb0
// CHECK_CAN:                   %2 = func.call @black_box_0() : () -> f32
// CHECK_CAN:                   cf.br ^bb3(%2 : f32)
// CHECK_CAN:               ^bb2:  // pred: ^bb0
// CHECK_CAN:                   %3 = func.call @black_box_1() : () -> f32
// CHECK_CAN:                   cf.br ^bb3(%3 : f32)
// CHECK_CAN:               ^bb3(%4: f32):  // 2 preds: ^bb1, ^bb2
// CHECK_CAN:                   scf.yield %4 : f32
// CHECK_CAN:               }
// CHECK_CAN:               scf.yield %1 : f32
// CHECK_CAN:           } else {
// CHECK_CAN:               scf.yield %arg2 : f32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> f32, sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> f32, sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i1, i1, f32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: i1, %arg2: f32):
// CHECK_GEN:           %0 = "scf.if"(%arg0) ({
// CHECK_GEN:               %1 = "scf.execute_region"() ({
// CHECK_GEN:                   "cf.cond_br"(%arg1)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// CHECK_GEN:               ^bb1:  // pred: ^bb0
// CHECK_GEN:                   %2 = "func.call"() <{callee = @black_box_0}> : () -> f32
// CHECK_GEN:                   "cf.br"(%2)[^bb3] : (f32) -> ()
// CHECK_GEN:               ^bb2:  // pred: ^bb0
// CHECK_GEN:                   %3 = "func.call"() <{callee = @black_box_1}> : () -> f32
// CHECK_GEN:                   "cf.br"(%3)[^bb3] : (f32) -> ()
// CHECK_GEN:               ^bb3(%4: f32):  // 2 preds: ^bb1, ^bb2
// CHECK_GEN:                   "scf.yield"(%4) : (f32) -> ()
// CHECK_GEN:               }) : () -> f32
// CHECK_GEN:               "scf.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               "scf.yield"(%arg2) : (f32) -> ()
// CHECK_GEN:           }) : (i1) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
