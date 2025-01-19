// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box_0(index) -> f32
    func.func private @black_box_1(f32, f32) -> f32
    func.func @test(%n: index, %N: index, %step: index, %init: f32) -> f32
    {
        %out = scf.parallel(%ind) = (%n) to (%N) step (%step) init (%init) -> f32 {
            %val = func.call @black_box_0(%ind) : (index) -> f32
            scf.reduce(%val: f32) {
            ^bb0(%lhs: f32, %rhs: f32):
                %res = func.call @black_box_1(%lhs, %rhs) : (f32, f32) -> f32
                scf.reduce.return %res : f32
            }
        }
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0(index) -> f32
// CHECK_CAN:       func.func private @black_box_1(f32, f32) -> f32
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> f32 {
// CHECK_CAN:           %0 = scf.parallel (%arg4) = (%arg0) to (%arg1) step (%arg2) init (%arg3) -> f32 {
// CHECK_CAN:               %1 = func.call @black_box_0(%arg4) : (index) -> f32
// CHECK_CAN:               scf.reduce(%1 : f32) {
// CHECK_CAN:               ^bb0(%arg5: f32, %arg6: f32):
// CHECK_CAN:                   %2 = func.call @black_box_1(%arg5, %arg6) : (f32, f32) -> f32
// CHECK_CAN:                   scf.reduce.return %2 : f32
// CHECK_CAN:               }
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> f32, sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (f32, f32) -> f32, sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, f32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: f32):
// CHECK_GEN:           %0 = "scf.parallel"(%arg0, %arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
// CHECK_GEN:           ^bb0(%arg4: index):
// CHECK_GEN:               %1 = "func.call"(%arg4) <{callee = @black_box_0}> : (index) -> f32
// CHECK_GEN:               "scf.reduce"(%1) ({
// CHECK_GEN:               ^bb0(%arg5: f32, %arg6: f32):
// CHECK_GEN:                   %2 = "func.call"(%arg5, %arg6) <{callee = @black_box_1}> : (f32, f32) -> f32
// CHECK_GEN:                   "scf.reduce.return"(%2) : (f32) -> ()
// CHECK_GEN:               }) : (f32) -> ()
// CHECK_GEN:           }) : (index, index, index, f32) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
