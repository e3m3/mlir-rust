// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test() -> tensor<4x3xf32>
    {
        %out = tensor.empty() : tensor<4x3xf32>
        func.return %out : tensor<4x3xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> tensor<4x3xf32> {
// CHECK_CAN:           %0 = tensor.empty() : tensor<4x3xf32>
// CHECK_CAN:           return %0 : tensor<4x3xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> tensor<4x3xf32>, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "tensor.empty"() : () -> tensor<4x3xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<4x3xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
