---

#   Issue: Vector TransferWrite Broadcast Dimension

Cannot legalize a `vector.transfer_write` operation as shown in the vector dialect documentation [1].

MLIR (`mlir-opt vector_transfer_write_5.mlir`):

```mlir
module {
    func.func @test(%base: tensor<f32>, %value: vector<1xf32>) -> tensor<f32>
    {
        %out = vector.transfer_write %value, %base[] {
            in_bounds = [true],
            permutation_map = affine_map<() -> (0)>
        } : vector<1xf32>, tensor<f32>
        func.return %out : tensor<f32>
    }
}
```

Error:

```
vector_transfer_write_5.mlir:19:16: error: 'vector.transfer_write' op should not have broadcast dimensions
        %out = vector.transfer_write %value, %base[] {
               ^
vector_transfer_write_5.mlir:19:16: note: see current operation: %0 = "vector.transfer_write"(%arg1, %arg0) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 0, 0>, permutation_map = affine_map<() -> (0)>}> : (vector<1xf32>, tensor<f32>) -> tensor<f32>
```

#   References

[1]:    https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_write-vectortransferwriteop

1.  `https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_write-vectortransferwriteop`
