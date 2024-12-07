// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<8x64xf32, affine_map<(d0,d1) -> (d0,d1)>, 1>) -> ()
    {
        memref.dealloc %mr0 : memref<8x64xf32, affine_map<(d0,d1) -> (d0,d1)>, 1>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<8x64xf32, 1>) {
// CHECK_CAN:           memref.dealloc %arg0 : memref<8x64xf32, 1>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<8x64xf32, 1>) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<8x64xf32, 1>):
// CHECK_GEN:           "memref.dealloc"(%arg0) : (memref<8x64xf32, 1>) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
