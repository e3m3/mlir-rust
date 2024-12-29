// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test(%in: i64) -> index
    {
        %out = index.castu %in : i64 to index
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: i64) -> index {
// CHECK_CAN:           %0 = index.castu %arg0 : i64 to index
// CHECK_CAN:           return %0 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (i64) -> index, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: i64):
// CHECK_GEN:           %0 = "index.castu"(%arg0) : (i64) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
