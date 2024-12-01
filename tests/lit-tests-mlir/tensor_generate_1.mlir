// RUN: mlir-opt -h                             | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN
// COM: [1]: https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  index
// CHECK-SAME:  tensor

module {
    func.func @test(%m : index, %n : index) -> tensor<?x3x?xf32>
        attributes { unused_attr = "test" }
    {
        %init = arith.constant 10.0 : f32

        %tnsr = tensor.generate %m, %n {
        ^bb0(%i : index, %j : index, %k : index):
            %j0 = index.casts %j : index to i32
            %j1 = arith.sitofp %j0 : i32 to f32
            %elem = arith.addf %init, %j1 : f32
            tensor.yield %elem : f32
        } : tensor<?x3x?xf32>

        return %tnsr : tensor<?x3x?xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index, %arg1: index) -> tensor<?x3x?xf32> attributes {unused_attr = "test"} {
// CHECK_CAN:           %cst = arith.constant 1.000000e+01 : f32
// CHECK_CAN:           %generated = tensor.generate %arg0, %arg1 {
// CHECK_CAN:           ^bb0(%arg2: index, %arg3: index, %arg4: index):
// CHECK_CAN:               %0 = index.casts %arg3 : index to i32
// CHECK_CAN:               %1 = arith.sitofp %0 : i32 to f32
// CHECK_CAN:               %2 = arith.addf %1, %cst : f32
// CHECK_CAN:               tensor.yield %2 : f32
// CHECK_CAN:           } : tensor<?x3x?xf32>
// CHECK_CAN:           return %generated : tensor<?x3x?xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index, index) -> tensor<?x3x?xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 1.000000e+01 : f32}> : () -> f32
// CHECK_GEN:           %1 = "tensor.generate"(%arg0, %arg1) ({
// CHECK_GEN:           ^bb0(%arg2: index, %arg3: index, %arg4: index):
// CHECK_GEN:               %2 = "index.casts"(%arg3) : (index) -> i32
// CHECK_GEN:               %3 = "arith.sitofp"(%2) : (i32) -> f32
// CHECK_GEN:               %4 = "arith.addf"(%0, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK_GEN:               "tensor.yield"(%4) : (f32) -> ()
// CHECK_GEN:           }) : (index, index) -> tensor<?x3x?xf32>
// CHECK_GEN:           "func.return"(%1) : (tensor<?x3x?xf32>) -> ()
// CHECK_GEN:       }) {unused_attr = "test"} : () -> ()
// CHECK_GEN:   }) : () -> ()
