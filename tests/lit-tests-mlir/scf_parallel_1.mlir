// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box(index) -> ()
    func.func @test(%n: index, %N: index, %step: index) -> ()
    {
        scf.parallel(%ind) = (%n) to (%N) step (%step) init () {
            func.call @black_box(%ind) : (index) -> ()
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box(index)
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index) {
// CHECK_CAN:           scf.parallel (%arg3) = (%arg0) to (%arg1) step (%arg2) {
// CHECK_CAN:               func.call @black_box(%arg3) : (index) -> ()
// CHECK_CAN:               scf.reduce
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> (), sym_name = "black_box", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index):
// CHECK_GEN:           "scf.parallel"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK_GEN:           ^bb0(%arg3: index):
// CHECK_GEN:               "func.call"(%arg3) <{callee = @black_box}> : (index) -> ()
// CHECK_GEN:               "scf.reduce"() : () -> ()
// CHECK_GEN:           }) : (index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
