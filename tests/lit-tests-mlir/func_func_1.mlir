// RUN: @mlir-opt -h                                                        | @filecheck %s
// RUN: @mlir-opt %s --canonicalize --allow-unregistered-dialect            | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect   | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  func

module {
    func.func private @test(i32 {lang.self = #lang.class<className> : index}, f64, f64) -> (f64 {dialectName.attrName = 1 : i32})
}

// CHECK_CAN:   module {
// CHECK_CAN:       func.func private @test(i32 {lang.self = #lang.class<className> : index}, f64, f64) -> (f64 {dialectName.attrName = 1 : i32})
// CHECK_CAN:   }

// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{arg_attrs = [{lang.self = #lang.class<className> : index}, {}, {}], function_type = (i32, f64, f64) -> f64, res_attrs = [{dialectName.attrName = 1 : i32}], sym_name = "test", sym_visibility = "private"}> ({
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
