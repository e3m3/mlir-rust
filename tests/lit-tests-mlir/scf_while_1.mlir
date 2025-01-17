// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  scf

module {
    func.func private @black_box_0() -> (i1)
    func.func private @black_box_1() -> ()
    func.func @test() -> ()
    {
        scf.while () : () -> () {
            %cond = func.call @black_box_0() : () -> (i1)
            scf.condition(%cond)
        } do {
            func.call @black_box_1() : () -> ()
            scf.yield
        }
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @black_box_0() -> i1
// CHECK_CAN:       func.func private @black_box_1()
// CHECK_CAN:       func.func @test() {
// CHECK_CAN:           scf.while : () -> () {
// CHECK_CAN:               %0 = func.call @black_box_0() : () -> i1
// CHECK_CAN:               scf.condition(%0)
// CHECK_CAN:           } do {
// CHECK_CAN:               func.call @black_box_1() : () -> ()
// CHECK_CAN:               scf.yield
// CHECK_CAN:           }
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> i1, sym_name = "black_box_0", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "black_box_1", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
// CHECK_GEN:           "scf.while"() ({
// CHECK_GEN:               %0 = "func.call"() <{callee = @black_box_0}> : () -> i1
// CHECK_GEN:               "scf.condition"(%0) : (i1) -> ()
// CHECK_GEN:           }, {
// CHECK_GEN:               "func.call"() <{callee = @black_box_1}> : () -> ()
// CHECK_GEN:               "scf.yield"() : () -> ()
// CHECK_GEN:           }) : () -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
