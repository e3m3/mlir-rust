// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func

module {
    func.func @test(%i: index, %c244: index) -> (index, index)
    {
        %out:2 = affine.delinearize_index %i into (%c244, %c244) : index, index
        func.return %out#0, %out#1 : index, index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> (index, index) {
// CHECK_CAN:           %0:2 = affine.delinearize_index %arg0 into (%arg1, %arg1) : index, index
// CHECK_CAN:           return %0#0, %0#1 : index, index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> (index, index), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0:2 = "affine.delinearize_index"(%arg0, %arg1, %arg1) : (index, index, index) -> (index, index)
// CHECK_GEN:           "func.return"(%0#0, %0#1) : (index, index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
