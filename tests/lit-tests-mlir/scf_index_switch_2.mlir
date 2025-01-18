// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box(index) -> (i32)
    func.func @test(%value: index, %default: i32) -> i32
    {
        %out = scf.index_switch %value -> i32
        case 2 {
            %res = func.call @black_box(%value) : (index) -> i32
            scf.yield %res : i32
        } case 5 {
            %res = func.call @black_box(%value) : (index) -> i32
            scf.yield %res : i32
        } default {
            scf.yield %default : i32
        }
        func.return %out : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index) -> i32
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: i32) -> i32 {
// CHECK_CAN:           %0 = scf.index_switch %arg0 -> i32
// CHECK_CAN:           case 2 {
// CHECK_CAN:               %1 = func.call @black_box(%arg0) : (index) -> i32
// CHECK_CAN:               scf.yield %1 : i32
// CHECK_CAN:           }
// CHECK_CAN:           case 5 {
// CHECK_CAN:               %1 = func.call @black_box(%arg0) : (index) -> i32
// CHECK_CAN:               scf.yield %1 : i32
// CHECK_CAN:           }
// CHECK_CAN:           default {
// CHECK_CAN:               scf.yield %arg1 : i32
// CHECK_CAN:           }
// CHECK_CAN:           return %0 : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> i32, sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, i32) -> i32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: i32):
// CHECK_GEN:               %0 = "scf.index_switch"(%arg0) <{cases = array<i64: 2, 5>}> ({
// CHECK_GEN:               "scf.yield"(%arg1) : (i32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               %2 = "func.call"(%arg0) <{callee = @black_box}> : (index) -> i32
// CHECK_GEN:               "scf.yield"(%2) : (i32) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               %1 = "func.call"(%arg0) <{callee = @black_box}> : (index) -> i32
// CHECK_GEN:               "scf.yield"(%1) : (i32) -> ()
// CHECK_GEN:           }) : (index) -> i32
// CHECK_GEN:           "func.return"(%0) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
