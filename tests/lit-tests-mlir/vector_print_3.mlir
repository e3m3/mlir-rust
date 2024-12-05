// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%v: vector<4xf32>) -> ()
    {
        vector.print %v : vector<4xf32> punctuation <comma>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<4xf32>) {
// CHECK_CAN:           vector.print %arg0 : vector<4xf32> punctuation <comma>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<4xf32>) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<4xf32>):
// CHECK_GEN:           "vector.print"(%arg0) <{punctuation = #vector.punctuation<comma>}> : (vector<4xf32>) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
