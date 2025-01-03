// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%in: vector<10xindex>) -> vector<10xi32>
    {
        %out = arith.index_cast %in : vector<10xindex> to vector<10xi32>
        func.return %out : vector<10xi32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<10xindex>) -> vector<10xi32> {
// CHECK_CAN:           %0 = arith.index_cast %arg0 : vector<10xindex> to vector<10xi32>
// CHECK_CAN:           return %0 : vector<10xi32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<10xindex>) -> vector<10xi32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<10xindex>):
// CHECK_GEN:           %0 = "arith.index_cast"(%arg0) : (vector<10xindex>) -> vector<10xi32>
// CHECK_GEN:           "func.return"(%0) : (vector<10xi32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()