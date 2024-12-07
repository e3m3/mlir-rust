// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%s: index) -> memref<8x64xf32, affine_map<(d0,d1)[s0] -> ((d0 + s0), d1)>>
    {
        %mr0 = memref.alloca()[%s] : memref<8x64xf32, affine_map<(d0,d1)[s0] -> ((d0 + s0), d1)>>
        func.return %mr0 : memref<8x64xf32, affine_map<(d0,d1)[s0] -> ((d0 + s0), d1)>>
    }
}

// CHECK_CAN:   #map = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index) -> memref<8x64xf32, #map> {
// CHECK_CAN:           %alloca = memref.alloca()[%arg0] : memref<8x64xf32, #map>
// CHECK_CAN:           return %alloca : memref<8x64xf32, #map>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (index) -> memref<8x64xf32, #map>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index):
// CHECK_GEN:           %0 = "memref.alloca"(%arg0) <{operandSegmentSizes = array<i32: 0, 1>}> : (index) -> memref<8x64xf32, #map>
// CHECK_GEN:           "func.return"(%0) : (memref<8x64xf32, #map>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
