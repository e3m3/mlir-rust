// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box() -> ()
    func.func @test(%cond: i1) -> ()
    {
        scf.if %cond {
            func.call @black_box() : () -> ()
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box()
// CHECK_CAN:       func.func @test(%arg0: i1) {
// CHECK_CAN:           scf.if %arg0 {
// CHECK_CAN:               func.call @black_box() : () -> ()
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (i1) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1):
// CHECK_GEN:           "scf.if"(%arg0) ({
// CHECK_GEN:               "func.call"() <{callee = @black_box}> : () -> ()
// CHECK_GEN:               "scf.yield"() : () -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:           }) : (i1) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
