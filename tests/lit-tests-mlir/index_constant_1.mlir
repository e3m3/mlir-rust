// RUN: @mlir-opt -h                            | @filecheck %s
// RUN: @mlir-opt %s --canonicalize             | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic    | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test() -> index
    {
        %out = index.constant 10
        func.return %out : index
    }
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test() -> index {
// CHECK_CAN:           %idx10 = index.constant 10
// CHECK_CAN:           return %idx10 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{function_type = () -> index, sym_name = "test"}> ({
// CHECK_GEN:           %0 = "index.constant"() <{value = 10 : index}> : () -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
