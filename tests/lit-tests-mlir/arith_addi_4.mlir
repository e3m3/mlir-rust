// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%a: vector<10xindex>, %b: vector<10xindex>) -> vector<10xindex>
    {
        %out = arith.addi %a, %b : vector<10xindex>
        func.return %out : vector<10xindex>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<10xindex>, %arg1: vector<10xindex>) -> vector<10xindex> {
// CHECK_CAN:           %0 = arith.addi %arg0, %arg1 : vector<10xindex>
// CHECK_CAN:           return %0 : vector<10xindex>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<10xindex>, vector<10xindex>) -> vector<10xindex>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<10xindex>, %arg1: vector<10xindex>):
// CHECK_GEN:           %0 = "arith.addi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (vector<10xindex>, vector<10xindex>) -> vector<10xindex>
// CHECK_GEN:           "func.return"(%0) : (vector<10xindex>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
