// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%a: index, %b: index) -> (index, i1)
    {
        %out, %overflow = arith.addui_extended %a, %b : index, i1
        func.return %out, %overflow : index, i1
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> (index, i1) {
// CHECK_CAN:           %sum, %overflow = arith.addui_extended %arg0, %arg1 : index, i1
// CHECK_CAN:           return %sum, %overflow : index, i1
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> (index, i1), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0:2 = "arith.addui_extended"(%arg0, %arg1) : (index, index) -> (index, i1)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (index, i1) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
