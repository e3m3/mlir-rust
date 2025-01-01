// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%a: vector<10xi8>, %b: vector<10xi8>) -> (vector<10xi8>, vector<10xi8>)
    {
        %low, %high = arith.mulsi_extended %a, %b : vector<10xi8>
        func.return %low, %high : vector<10xi8>, vector<10xi8>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<10xi8>, %arg1: vector<10xi8>) -> (vector<10xi8>, vector<10xi8>) {
// CHECK_CAN:           %low, %high = arith.mulsi_extended %arg0, %arg1 : vector<10xi8>
// CHECK_CAN:           return %low, %high : vector<10xi8>, vector<10xi8>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<10xi8>, vector<10xi8>) -> (vector<10xi8>, vector<10xi8>), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<10xi8>, %arg1: vector<10xi8>):
// CHECK_GEN:           %0:2 = "arith.mulsi_extended"(%arg0, %arg1) : (vector<10xi8>, vector<10xi8>) -> (vector<10xi8>, vector<10xi8>)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (vector<10xi8>, vector<10xi8>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
