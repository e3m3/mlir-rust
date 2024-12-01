// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  arith
// CHECK-SAME:  func
// CHECK-SAME:  vector

module {
    func.func @test() -> ()
    {
        %a = arith.constant 1.0 : f32
        %b = arith.constant 2.0 : f32
        %c = arith.constant 3.0 : f32
        %d = arith.constant 4.0 : f32
        %v = vector.from_elements %a, %b, %c, %d : vector<4xf32>
        vector.print %v : vector<4xf32> punctuation <comma>
        func.return
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() {
// CHECK_CAN:           %cst = arith.constant 1.000000e+00 : f32
// CHECK_CAN:           %cst_0 = arith.constant 2.000000e+00 : f32
// CHECK_CAN:           %cst_1 = arith.constant 3.000000e+00 : f32
// CHECK_CAN:           %cst_2 = arith.constant 4.000000e+00 : f32
// CHECK_CAN:           %0 = vector.from_elements %cst, %cst_0, %cst_1, %cst_2 : vector<4xf32>
// CHECK_CAN:           vector.print %0 : vector<4xf32> punctuation <comma>
// CHECK_CAN:           return
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
// CHECK_GEN:           %0 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
// CHECK_GEN:           %1 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
// CHECK_GEN:           %2 = "arith.constant"() <{value = 3.000000e+00 : f32}> : () -> f32
// CHECK_GEN:           %3 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
// CHECK_GEN:           %4 = "vector.from_elements"(%0, %1, %2, %3) : (f32, f32, f32, f32) -> vector<4xf32>
// CHECK_GEN:           "vector.print"(%4) <{punctuation = #vector.punctuation<comma>}> : (vector<4xf32>) -> ()
// CHECK_GEN:           "func.return"() : () -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
