

PETSc version 3.22.x

Ensure the `PETSC_DIR` environment variable is set.
If you are using a local build, you probably need to set `PETSC_ARCH`.

Compile via

```
make -f Makefile.basic.user ex_ortho
```

or compile using the PETSc makefile 

```
make -f $PETSC_DIR/share/petsc/Makefile.basic.user ex_ortho
```

All C files can be compiled the same way.

