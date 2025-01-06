// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xi1>
    {
        %out = arith.cmpf oeq, %a, %b : vector<4xf32>
        func.return %out : vector<4xi1>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xi1> {
// CHECK_CAN:           %0 = arith.cmpf oeq, %arg0, %arg1 : vector<4xf32>
// CHECK_CAN:           return %0 : vector<4xi1>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<4xf32>, vector<4xf32>) -> vector<4xi1>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<4xf32>, %arg1: vector<4xf32>):
// CHECK_GEN:           %0 = "arith.cmpf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xi1>
// CHECK_GEN:           "func.return"(%0) : (vector<4xi1>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
