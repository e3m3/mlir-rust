// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box(index) -> ()
    func.func @test(%value: index) -> ()
    {
        scf.index_switch %value
        case 2 {
            func.call @black_box(%value) : (index) -> ()
            scf.yield
        } case 5 {
            func.call @black_box(%value) : (index) -> ()
            scf.yield
        } default {
            scf.yield
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index)
// CHECK_CAN:       func.func @test(%arg0: index) {
// CHECK_CAN:           scf.index_switch %arg0
// CHECK_CAN:           case 2 {
// CHECK_CAN:               func.call @black_box(%arg0) : (index) -> ()
// CHECK_CAN:               scf.yield
// CHECK_CAN:           }
// CHECK_CAN:           case 5 {
// CHECK_CAN:               func.call @black_box(%arg0) : (index) -> ()
// CHECK_CAN:               scf.yield
// CHECK_CAN:           }
// CHECK_CAN:           default {
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index):
// CHECK_GEN:           "scf.index_switch"(%arg0) <{cases = array<i64: 2, 5>}> ({
// CHECK_GEN:               "scf.yield"() : () -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               "func.call"(%arg0) <{callee = @black_box}> : (index) -> ()
// CHECK_GEN:               "scf.yield"() : () -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               "func.call"(%arg0) <{callee = @black_box}> : (index) -> ()
// CHECK_GEN:               "scf.yield"() : () -> ()
// CHECK_GEN:           }) : (index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
