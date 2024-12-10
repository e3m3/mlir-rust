// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%t: tensor<?x32xf32>, %m: index, %n: index) -> tensor<?x?x32xf32>
    {
        %out = tensor.expand_shape %t [[0,1], [2]] output_shape[%m, %n, 32] :
            tensor<?x32xf32> into tensor<?x?x32xf32>
        func.return %out : tensor<?x?x32xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: tensor<?x32xf32>, %arg1: index, %arg2: index) -> tensor<?x?x32xf32> {
// CHECK_CAN:           %expanded = tensor.expand_shape %arg0 [[DBL_SQ_BKT_L]]0, 1], [2[[DBL_SQ_BKT_R]] output_shape [%arg1, %arg2, 32] : tensor<?x32xf32> into tensor<?x?x32xf32>
// CHECK_CAN:           return %expanded : tensor<?x?x32xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (tensor<?x32xf32>, index, index) -> tensor<?x?x32xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: tensor<?x32xf32>, %arg1: index, %arg2: index):
// CHECK_GEN:           %0 = "tensor.expand_shape"(%arg0, %arg1, %arg2) <{reassociation = [[DBL_SQ_BKT_L]]0, 1], [2[[DBL_SQ_BKT_R]], static_output_shape = array<i64: -9223372036854775808, -9223372036854775808, 32>}> : (tensor<?x32xf32>, index, index) -> tensor<?x?x32xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<?x?x32xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
