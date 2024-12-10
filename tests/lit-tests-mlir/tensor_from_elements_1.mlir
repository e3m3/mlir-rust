// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(
        %a: index,
        %b: index,
        %c: index,
        %d: index,
        %e: index,
        %f: index
    ) -> tensor<2x3xindex> {
        %out = tensor.from_elements %a, %b, %c, %d, %e, %f : tensor<2x3xindex>
        func.return %out : tensor<2x3xindex>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<2x3xindex> {
// CHECK_CAN:           %from_elements = tensor.from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<2x3xindex>
// CHECK_CAN:           return %from_elements : tensor<2x3xindex>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index, index, index, index, index) -> tensor<2x3xindex>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
// CHECK_GEN:           %0 = "tensor.from_elements"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (index, index, index, index, index, index) -> tensor<2x3xindex>
// CHECK_GEN:           "func.return"(%0) : (tensor<2x3xindex>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
