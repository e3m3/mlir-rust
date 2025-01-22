// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box(index, f32) -> f32
    func.func @test(%n: index, %N: index, %step: index, %init: f32) -> f32
    {
        %out = scf.for %ind = %n to %N step %step iter_args(%acc = %init) -> f32 {
            %res = scf.execute_region -> f32 {
                %value = func.call @black_box(%ind, %acc) : (index, f32) -> f32
                scf.yield %value : f32
            }
            scf.yield %res : f32
        }
        func.return %out : f32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index, f32) -> f32
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> f32 {
// CHECK_CAN:           %0 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %arg3) -> (f32) {
// CHECK_CAN:               %1 = func.call @black_box(%arg4, %arg5) : (index, f32) -> f32
// CHECK_CAN:               scf.yield %1 : f32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : f32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, f32) -> f32, sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, f32) -> f32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: f32):
// CHECK_GEN:           %0 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
// CHECK_GEN:           ^bb0(%arg4: index, %arg5: f32):
// CHECK_GEN:               %1 = "scf.execute_region"() ({
// CHECK_GEN:                   %2 = "func.call"(%arg4, %arg5) <{callee = @black_box}> : (index, f32) -> f32
// CHECK_GEN:                   "scf.yield"(%2) : (f32) -> ()
// CHECK_GEN:               }) : () -> f32
// CHECK_GEN:               "scf.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }) : (index, index, index, f32) -> f32
// CHECK_GEN:           "func.return"(%0) : (f32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
