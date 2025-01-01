// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func

module {
    func.func @test(%in: i32) -> index
    {
        %out = arith.index_castui %in : i32 to index
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i32) -> index {
// CHECK_CAN:           %0 = arith.index_castui %arg0 : i32 to index
// CHECK_CAN:           return %0 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i32) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i32):
// CHECK_GEN:           %0 = "arith.index_castui"(%arg0) : (i32) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
