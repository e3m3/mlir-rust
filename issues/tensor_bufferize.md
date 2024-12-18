---

#   Issue: Linalg Matmul Cast Attribute

Program:

```mlir
module {
    func.func @test(%t: tensor<4x4x4xf32>, %indices: tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
    {
        %out = tensor.gather %t[%indices] gather_dims([0,1,2]) :
            (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
        func.return %out : tensor<1x2x1x1x1xf32>
    }
}
```

Output (`mlir-opt %s --canonicalize --sparsification-and-bufferization`):

```mlir
module {
    func.func @test(%arg0: memref<4x4x4xf32>, %arg1: memref<1x2x3xindex>) -> memref<1x2x1x1x1xf32> {
        %0 = bufferization.to_tensor %arg1 : memref<1x2x3xindex>
        %1 = bufferization.to_tensor %arg0 : memref<4x4x4xf32>
        %gather = tensor.gather %1[%0] gather_dims([0, 1, 2]) : (tensor<4x4x4xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
        %2 = bufferization.to_memref %gather : memref<1x2x1x1x1xf32>
        return %2 : memref<1x2x1x1x1xf32>
    }
}
```

Output (`mlir-opt %s --canonicalize --sparsification-and-bufferization --one-shot-bufferize`):

```
tensor_gather_1.mlir:10:44: error: 'bufferization.to_tensor' op to_tensor ops without `restrict` are not supported by One-Shot Analysis
    func.func @test(%t: tensor<4x4x4xf32>, %indices: tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
                                           ^
tensor_gather_1.mlir:10:44: note: see current operation: %0 = "bufferization.to_tensor"(%arg1) : (memref<1x2x3xindex>) -> tensor<1x2x3xindex>
```

Output (`mlir-opt %s --canonicalize --sparsification-and-bufferization --sparsifier`):

```
tensor_gather_1.mlir:10:44: error: 'bufferization.to_tensor' op to_tensor ops without `restrict` are not supported by One-Shot Analysis
    func.func @test(%t: tensor<4x4x4xf32>, %indices: tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
                                           ^
tensor_gather_1.mlir:10:44: note: see current operation: %0 = "bufferization.to_tensor"(%arg1) : (memref<1x2x3xindex>) -> tensor<1x2x3xindex>
```


#   References

[1]:    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop

1.  `https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop`
