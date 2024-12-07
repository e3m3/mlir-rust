// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  memref

module {
    func.func @test(%mr0: memref<?x?xf32>) -> memref<?x?xf32, affine_map<(d0,d1)[s0] -> (d1*s0 + d0)>>
    {
        %out = memref.transpose %mr0 (i,j) -> (j,i) : memref<?x?xf32>
            to memref<?x?xf32, affine_map<(d0,d1)[s0] -> (d1*s0 + d0)>>
        func.return %out : memref<?x?xf32, affine_map<(d0,d1)[s0] -> (d1*s0 + d0)>>
    }
}

// CHECK_CAN:   #map = affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: memref<?x?xf32>) -> memref<?x?xf32, #map> {
// CHECK_CAN:           %transpose = memref.transpose %arg0 (d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, #map>
// CHECK_CAN:           return %transpose : memref<?x?xf32, #map>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>
// CHECK_GEN:   #map1 = affine_map<(d0, d1) -> (d1, d0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (memref<?x?xf32>) -> memref<?x?xf32, #map>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: memref<?x?xf32>):
// CHECK_GEN:           %0 = "memref.transpose"(%arg0) <{permutation = #map1}> : (memref<?x?xf32>) -> memref<?x?xf32, #map>
// CHECK_GEN:           "func.return"(%0) : (memref<?x?xf32, #map>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
