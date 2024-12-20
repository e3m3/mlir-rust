// RUN: mlir-opt -h                             | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN
// COM: [1]: https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  tensor

module {
    func.func @test(%m : index, %n : index) -> tensor<?x3x?xf32>
    {
        %tnsr = tensor.generate %m, %n {
        ^bb0(%i : index, %j : index, %k : index):
            %elem = arith.constant 0.0 : f32
            tensor.yield %elem : f32
        } : tensor<?x3x?xf32>
        return %tnsr : tensor<?x3x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> tensor<?x3x?xf32> {
// CHECK_CAN:           %cst = arith.constant 0.000000e+00 : f32
// CHECK_CAN:           %generated = tensor.generate %arg0, %arg1 {
// CHECK_CAN:           ^bb0(%arg2: index, %arg3: index, %arg4: index):
// CHECK_CAN:               tensor.yield %cst : f32
// CHECK_CAN:           } : tensor<?x3x?xf32>
// CHECK_CAN:           return %generated : tensor<?x3x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> tensor<?x3x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "tensor.generate"(%arg0, %arg1) ({
// CHECK_GEN:           ^bb0(%arg2: index, %arg3: index, %arg4: index):
// CHECK_GEN:               %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CHECK_GEN:               "tensor.yield"(%1) : (f32) -> ()
// CHECK_GEN:           }) : (index, index) -> tensor<?x3x?xf32>
// CHECK_GEN:           "func.return"(%0) : (tensor<?x3x?xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
