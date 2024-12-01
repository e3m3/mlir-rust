// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test(%v: vector<4x8x16xf32>) -> vector<16xf32>
    {
        %a = arith.constant 7 : index
        %v0 = vector.extract %v[2, %a] : vector<16xf32> from vector<4x8x16xf32>
        func.return %v0 : vector<16xf32>
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: vector<4x8x16xf32>) -> vector<16xf32> {
// CHECK_CAN:           %c7 = arith.constant 7 : index
// CHECK_CAN:           %0 = vector.extract %arg0[2, %c7] : vector<16xf32> from vector<4x8x16xf32>
// CHECK_CAN:           return %0 : vector<16xf32>
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = (vector<4x8x16xf32>) -> vector<16xf32>, sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: vector<4x8x16xf32>):
// CHECK_GEN:           %0 = "arith.constant"() <{value = 7 : index}> : () -> index
// CHECK_GEN:           %1 = "vector.extract"(%arg0, %0) <{static_position = array<i64: 2, -9223372036854775808>}> : (vector<4x8x16xf32>, index) -> vector<16xf32>
// CHECK_GEN:           "func.return"(%1) : (vector<16xf32>) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
