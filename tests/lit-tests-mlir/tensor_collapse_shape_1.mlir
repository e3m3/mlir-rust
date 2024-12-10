// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<?x?x?xf32>) -> tensor<?x?xf32>
    {
        %t0 = tensor.collapse_shape %t [[0,1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
        func.return %t0 : tensor<?x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
// CHECK_CAN:           %collapsed = tensor.collapse_shape %arg0 [[DBL_SQ_BKT_L]]0, 1], [2[[DBL_SQ_BKT_R]] : tensor<?x?x?xf32> into tensor<?x?xf32>
// CHECK_CAN:           return %collapsed : tensor<?x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<?x?x?xf32>) -> tensor<?x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<?x?x?xf32>):
// CHECK_GEN:           %0 = "tensor.collapse_shape"(%arg0) <{reassociation = [[DBL_SQ_BKT_L]]0, 1], [2[[DBL_SQ_BKT_R]]}> : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<?x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
