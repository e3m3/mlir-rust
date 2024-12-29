// RUN: @mlir-opt -h                                                        | @filecheck %s
// RUN: @mlir-opt %s --canonicalize --allow-unregistered-dialect            | @filecheck %s --check-prefix=CHECK_CAN
// RUN: @mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect   | @filecheck %s --check-prefix=CHECK_GEN

// CHECK:       Available Dialects:
// CHECK-SAME:  affine
// CHECK-SAME:  func
// CHECK-SAME:  index

module {
    func.func @test(
        %i: index {hoare.constraints = affine_set<(d0): (d0 >= 0)>},
        %j: index {hoare.constraints = affine_set<(d1): (d1 >= 0)>}
    ) -> (index {hoare.constraints = affine_set<(d0,d1): (d0 >= 0, d1 >= 0, d0 + d1 >= 0)>}) {
        %out = index.add %i, %j
        func.return %out : index
    }
}

// CHECK_CAN:   #set = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 + d1 >= 0)>
// CHECK_CAN:   module {
// CHECK_CAN:       func.func @test(%arg0: index {hoare.constraints = affine_set<(d0) : (d0 >= 0)>}, %arg1: index {hoare.constraints = affine_set<(d0) : (d0 >= 0)>}) -> (index {hoare.constraints = #set}) {
// CHECK_CAN:           %0 = index.add %arg0, %arg1
// CHECK_CAN:           return %0 : index
// CHECK_CAN:       }
// CHECK_CAN:   }

// CHECK_GEN:   #set = affine_set<(d0) : (d0 >= 0)>
// CHECK_GEN:   #set1 = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 + d1 >= 0)>
// CHECK_GEN:   "builtin.module"() ({
// CHECK_GEN:       "func.func"() <{arg_attrs = [{hoare.constraints = #set}, {hoare.constraints = #set}], function_type = (index, index) -> index, res_attrs = [{hoare.constraints = #set1}], sym_name = "test"}> ({
// CHECK_GEN:       ^bb0(%arg0: index, %arg1: index):
// CHECK_GEN:           %0 = "index.add"(%arg0, %arg1) : (index, index) -> index
// CHECK_GEN:           "func.return"(%0) : (index) -> ()
// CHECK_GEN:       }) : () -> ()
// CHECK_GEN:   }) : () -> ()
