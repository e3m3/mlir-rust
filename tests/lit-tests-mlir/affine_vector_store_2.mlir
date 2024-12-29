// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  memref
// CHECK-SAME:  vector

module {
    func.func @test(
        %value: vector<8xf32>,
        %base: memref<100x100xf32>,
        %i: index,
        %j: index,
        %n: index,
        %m: index
    ) -> () {
        affine.vector_store %value, %base[%i + symbol(%n), %j + symbol(%m)]
            : memref<100x100xf32>, vector<8xf32>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<8xf32>, %arg1: memref<100x100xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
// CHECK_CAN:           affine.vector_store %arg0, %arg1[symbol(%arg4) + symbol(%arg2), symbol(%arg5) + symbol(%arg3)] : memref<100x100xf32>, vector<8xf32>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #map = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<8xf32>, memref<100x100xf32>, index, index, index, index) -> (), sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<8xf32>, %arg1: memref<100x100xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
// CHECK_GEN:           "affine.vector_store"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{map = #map}> : (vector<8xf32>, memref<100x100xf32>, index, index, index, index) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
