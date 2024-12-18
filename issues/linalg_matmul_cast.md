---

#   Issue: Linalg Matmul Cast Attribute

Program:

```mlir
module {
    func.func @test(%a: memref<3x5xi64>, %b: memref<5x7xi64>) -> memref<3x7xi64>
    {
        %out = memref.alloc() : memref<3x7xi64>
        linalg.matmul {cast = 0 : i32}
            ins(%a, %b: memref<3x5xi64>, memref<5x7xi64>) outs(%out: memref<3x7xi64>)
        func.return %out : memref<3x7xi64>
    }
}
```

Output (`mlir-opt %s --canonicalize`):

```
linalg_matmul_5.mlir:15:23: error: custom op 'linalg.matmul' 'linalg.matmul' op attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1
        linalg.matmul {cast = 0 : i32}
```


#   References

[1]:    https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop

1.  `https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop`
