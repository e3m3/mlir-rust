// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%a: i1, %b: i1, %c: i1, %d: i1) -> vector<4xi1>
    {
        %v = vector.from_elements %a, %b, %c, %d : vector<4xi1>
        func.return %v : vector<4xi1>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> vector<4xi1> {
// CHECK_CAN:           %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<4xi1>
// CHECK_CAN:           return %0 : vector<4xi1>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i1, i1, i1, i1) -> vector<4xi1>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1):
// CHECK_GEN:           %0 = "vector.from_elements"(%arg0, %arg1, %arg2, %arg3) : (i1, i1, i1, i1) -> vector<4xi1>
// CHECK_GEN:           "func.return"(%0) : (vector<4xi1>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
