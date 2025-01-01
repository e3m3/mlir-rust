// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%in: index) -> i32
    {
        %out = arith.index_castui %in : index to i32
        func.return %out : i32
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index) -> i32 {
// CHECK_CAN:           %0 = arith.index_castui %arg0 : index to i32
// CHECK_CAN:           return %0 : i32
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> i32, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index):
// CHECK_GEN:           %0 = "arith.index_castui"(%arg0) : (index) -> i32
// CHECK_GEN:           "func.return"(%0) : (i32) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
