// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%in: vector<10xi64>) -> vector<10xf64>
    {
        %out = arith.uitofp %in : vector<10xi64> to vector<10xf64>
        func.return %out : vector<10xf64>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<10xi64>) -> vector<10xf64> {
// CHECK_CAN:           %0 = arith.uitofp %arg0 : vector<10xi64> to vector<10xf64>
// CHECK_CAN:           return %0 : vector<10xf64>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<10xi64>) -> vector<10xf64>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<10xi64>):
// CHECK_GEN:           %0 = "arith.uitofp"(%arg0) : (vector<10xi64>) -> vector<10xf64>
// CHECK_GEN:           "func.return"(%0) : (vector<10xf64>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
