// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box_0(index) -> (f32, f32)
    func.func private @black_box_1(f32, f32) -> f32
    func.func private @black_box_2(f32, f32) -> f32
    func.func @test(%n: index, %N: index, %step: index, %init_lhs: f32, %init_rhs: f32) -> (f32, f32)
    {
        %out:2 = scf.parallel(%ind) = (%n) to (%N) step (%step) init (%init_lhs, %init_rhs) -> (f32, f32) {
            %x, %y = func.call @black_box_0(%ind) : (index) -> (f32, f32)
            scf.reduce(%x, %y: f32, f32) {
            ^bb0(%lhs: f32, %rhs: f32):
                %res = func.call @black_box_1(%lhs, %rhs) : (f32, f32) -> f32
                scf.reduce.return %res : f32
            }, {
            ^bb0(%lhs: f32, %rhs: f32):
                %res = func.call @black_box_2(%lhs, %rhs) : (f32, f32) -> f32
                scf.reduce.return %res : f32
            }
        }
        func.return %out#0, %out#1 : f32, f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(index) -> (f32, f32)
// CHECK_CAN:       func.func private @black_box_1(f32, f32) -> f32
// CHECK_CAN:       func.func private @black_box_2(f32, f32) -> f32
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: f32, %arg4: f32) -> (f32, f32) {
// CHECK_CAN:           %0:2 = scf.parallel (%arg5) = (%arg0) to (%arg1) step (%arg2) init (%arg3, %arg4) -> (f32, f32) {
// CHECK_CAN:               %1:2 = func.call @black_box_0(%arg5) : (index) -> (f32, f32)
// CHECK_CAN:               scf.reduce(%1#0, %1#1 : f32, f32) {
// CHECK_CAN:               ^bb0(%arg6: f32, %arg7: f32):
// CHECK_CAN:                   %2 = func.call @black_box_1(%arg6, %arg7) : (f32, f32) -> f32
// CHECK_CAN:                   scf.reduce.return %2 : f32
// CHECK_CAN:               }, {
// CHECK_CAN:               ^bb0(%arg6: f32, %arg7: f32):
// CHECK_CAN:                   %2 = func.call @black_box_2(%arg6, %arg7) : (f32, f32) -> f32
// CHECK_CAN:                   scf.reduce.return %2 : f32
// CHECK_CAN:               }
// CHECK_CAN:           }
// CHECK_CAN:           return %0#0, %0#1 : f32, f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (f32, f32), sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32) -> f32, sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32) -> f32, sym_name = "black_box_2", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, f32, f32) -> (f32, f32), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: f32, %arg4: f32):
// CHECK_GEN:           %0:2 = "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK_GEN:           ^bb0(%arg5: index):
// CHECK_GEN:               %1:2 = "func.call"(%arg5) <{callee = @black_box_0}> : (index) -> (f32, f32)
// CHECK_GEN:               "scf.reduce"(%1#0, %1#1) ({
// CHECK_GEN:               ^bb0(%arg8: f32, %arg9: f32):
// CHECK_GEN:                   %3 = "func.call"(%arg8, %arg9) <{callee = @black_box_1}> : (f32, f32) -> f32
// CHECK_GEN:                   "scf.reduce.return"(%3) : (f32) -> ()
// CHECK_GEN:               }, {
// CHECK_GEN:               ^bb0(%arg6: f32, %arg7: f32):
// CHECK_GEN:                   %2 = "func.call"(%arg6, %arg7) <{callee = @black_box_2}> : (f32, f32) -> f32
// CHECK_GEN:                   "scf.reduce.return"(%2) : (f32) -> ()
// CHECK_GEN:               }) : (f32, f32) -> ()
// CHECK_GEN:           }) : (index, index, index, f32, f32) -> (f32, f32)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (f32, f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
