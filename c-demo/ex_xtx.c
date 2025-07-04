static char help[] = "X^TX demo.";

#include <petsc.h>

// naive implementation
PetscErrorCode MatMatMultQRInplace_SEQDENSE_naive(Mat X,Mat R)
{
  Mat Rt;
  Vec q,row;
  PetscInt i,j,m,n,mr,nr;
  PetscScalar *_q,*_row,*_array;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(R,&mr,&nr));
  if (mr != nr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"R.shape[0] != R.shape[1]");
  PetscCall(MatGetSize(X,&m,&n));
  if (n != mr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Q.shape[1] != R.shape[0]");

  PetscCall(MatTranspose(R,MAT_INITIAL_MATRIX,&Rt));
  PetscCall(PetscCalloc1(mr,&_q));
  PetscCall(PetscCalloc1(mr,&_row));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,mr,_q,&q));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,mr,_row,&row));

  PetscCall(MatDenseGetArray(X,&_array));
  for (i=0; i<m; i++) {
    // get row q[i,:] = Q[i,:]
    for (j=0; j<n; j++) {
      _q[j] = _array[i + j*m];
    }
    PetscCall(MatMult(Rt,q,row));
    // insert row Q[i,:] = row[:]
    for (j=0; j<n; j++) {
       _array[i + j*m] = _row[j];
    }
  }
  PetscCall(MatDenseRestoreArray(X,&_array));
  PetscCall(PetscFree(_q));
  PetscCall(PetscFree(_row));
  PetscCall(VecDestroy(&q));
  PetscCall(VecDestroy(&row));
  PetscCall(MatDestroy(&Rt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// use column pointers in the hope to reduce bad memory access
PetscErrorCode MatMatMultQRInplace_SEQDENSE_memaccess(Mat X,Mat R)
{
  Mat Rt;
  Vec q,row;
  PetscInt i,j,m,n,mr,nr;
  PetscScalar *_q,*_row,*_array;
  PetscScalar **_a;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(R,&mr,&nr));
  if (mr != nr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"R.shape[0] != R.shape[1]");
  PetscCall(MatGetSize(X,&m,&n));
  if (n != mr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Q.shape[1] != R.shape[0]");

  PetscCall(MatTranspose(R,MAT_INITIAL_MATRIX,&Rt));
  PetscCall(PetscCalloc1(mr,&_q));
  PetscCall(PetscCalloc1(mr,&_row));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,mr,_q,&q));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,mr,_row,&row));

  //MatView(X,PETSC_VIEWER_STDOUT_SELF);
  PetscCall(MatDenseGetArray(X,&_array));
  PetscCall(PetscCalloc1(mr,&_a));
  for (j=0; j<n; j++) {
    _a[j] = &_array[0 + j*m];
  }

  for (i=0; i<m; i++) {
    // get row q[i,:] = Q[i,:]
    for (j=0; j<n; j++) {
      _q[j] = *_a[j];
    }
    PetscCall(MatMult(Rt,q,row));
    // insert row Q[i,:] = row[:]
    for (j=0; j<n; j++) {
      //PetscScalar *col = _a[j];
      *_a[j] = _row[j];
      _a[j]++;
    }
  }
  PetscCall(MatDenseRestoreArray(X,&_array));
  PetscCall(PetscFree(_a));
  PetscCall(PetscFree(_q));
  PetscCall(PetscFree(_row));
  PetscCall(VecDestroy(&q));
  PetscCall(VecDestroy(&row));
  PetscCall(MatDestroy(&Rt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// blocked variant
PetscErrorCode MatMatMultQRInplace_SEQDENSE_blocked_1(Mat X,Mat R)
{
  Mat bufX,bufXR;
  PetscInt i,j,m,n,mr,nr,bs;
  PetscScalar *_array;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(R,&mr,&nr));
  if (mr != nr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"R.shape[0] != R.shape[1]");
  PetscCall(MatGetSize(X,&m,&n));
  if (n != mr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Q.shape[1] != R.shape[0]");

  PetscCall(MatDuplicate(R,MAT_DO_NOT_COPY_VALUES,&bufX));
  PetscCall(MatDuplicate(R,MAT_DO_NOT_COPY_VALUES,&bufXR));

  PetscCall(MatDenseGetArray(X,&_array));

  bs = m/mr;
  if (bs * mr < m) {
    bs++;
  }
  for (i=0; i<bs; i++) {
    PetscInt ri,start,end,row = i*mr;
    PetscScalar *_x,*_xr;
    PetscBool trunc = PETSC_FALSE;

    start = row;
    end   = start + mr;
    if (end > m) {
      end = m;
      trunc = PETSC_TRUE;
    }
    //printf("i %d - start/end %d/%d\n",i,start,end);
    //if (start + bs > m) break;
    // fill
    PetscCall(MatDenseGetArray(bufX,&_x));
    if (trunc) { PetscMemzero(_x,mr*mr*sizeof(PetscScalar)); }
    for (j=0; j<mr; j++) {
      for (ri=start; ri<end; ri++) {
        _x[ri-start + j*mr] = _array[ri + j*m];
      }
    }
    PetscCall(MatDenseRestoreArray(bufX,&_x));

    // matmatmult
    PetscCall(MatMatMult(bufX,R,MAT_REUSE_MATRIX,1.0,&bufXR));

    // unpack
    PetscCall(MatDenseGetArray(bufXR,&_xr));
    for (j=0; j<mr; j++) {
      for (ri=start; ri<end; ri++) {
        _array[ri + j*m] = _xr[ri-start + j*mr];
      }
    }
    PetscCall(MatDenseRestoreArray(bufXR,&_xr));
  }

  PetscCall(MatDenseRestoreArray(X,&_array));
  PetscCall(MatDestroy(&bufX));
  PetscCall(MatDestroy(&bufXR));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultQRInplace_SEQDENSE_blocked_2(Mat X,Mat R)
{
  Mat bufX,bufXR;
  PetscInt i,j,m,n,mr,nr,fact,chunks,vector_length;
  PetscScalar *_array;

  PetscFunctionBeginUser;
  fact = 1;
  PetscOptionsGetInt(NULL,NULL,"-bs_mult",&fact,NULL);
  PetscCall(MatGetSize(R,&mr,&nr));
  if (mr != nr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"R.shape[0] != R.shape[1]");
  PetscCall(MatGetSize(X,&m,&n));
  if (n != mr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Q.shape[1] != R.shape[0]");

  PetscCall(MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, fact*mr, nr, NULL, &bufX));
  PetscCall(MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, fact*mr, nr, NULL, &bufXR));

  PetscCall(MatDenseGetArray(X,&_array));

  vector_length = fact*mr;
  chunks = m/vector_length;
  if (chunks * vector_length < m) {
    chunks++;
  }
  for (i=0; i<chunks; i++) {
    PetscInt ri,start,end,row = i*vector_length;
    PetscScalar *_x,*_xr;
    PetscBool trunc = PETSC_FALSE;

    start = row;
    end   = start + vector_length;
    if (end > m) {
      end = m;
      trunc = PETSC_TRUE;
    }
    //printf("i %d - start/end %d/%d\n",i,start,end);
    //if (start + bs > m) break;
    // fill
    PetscCall(MatDenseGetArray(bufX,&_x));
    if (trunc) { PetscMemzero(_x,vector_length*mr*sizeof(PetscScalar)); }
    for (j=0; j<mr; j++) {
      for (ri=start; ri<end; ri++) {
        _x[ri-start + j*vector_length] = _array[ri + j*m];
      }
    }
    PetscCall(MatDenseRestoreArray(bufX,&_x));

    // matmatmult
    PetscCall(MatMatMult(bufX,R,MAT_REUSE_MATRIX,1.0,&bufXR));

    // unpack
    PetscCall(MatDenseGetArray(bufXR,&_xr));
    for (j=0; j<mr; j++) {
      for (ri=start; ri<end; ri++) {
        _array[ri + j*m] = _xr[ri-start + j*vector_length];
      }
    }
    PetscCall(MatDenseRestoreArray(bufXR,&_xr));
  }

  PetscCall(MatDenseRestoreArray(X,&_array));
  PetscCall(MatDestroy(&bufX));
  PetscCall(MatDestroy(&bufXR));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultQRInplace_MPIDENSE(Mat Q,Mat R)
{
  Mat Ql;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetLocalMatrix(Q,&Ql));
  PetscCall(MatMatMultQRInplace_SEQDENSE_blocked_1(Ql,R));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultQRInplace_DENSE(Mat Q,Mat R)
{
  PetscMPIInt commsize;

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_DEBUG)
    {
      PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
      PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense[0]);
      PetscObjectTypeCompare((PetscObject)Q,MATMPIDENSE,&isdense[1]);
      if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid if Q is MatType MATSEQDENSE or MATMPIDENSE");

      PetscObjectTypeCompare((PetscObject)R,MATSEQDENSE,&isdense[0]);
      if (!isdense[0]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid if R is MatType MATSEQDENSE");
    }
  #endif
  MPI_Comm_size(PetscObjectComm((PetscObject)Q),&commsize);
  if (commsize == 1) {
    //PetscCall(MatMatMultQRInplace_SEQDENSE_naive(Q,R));
    //PetscCall(MatMatMultQRInplace_SEQDENSE_memaccess(Q,R));
    PetscCall(MatMatMultQRInplace_SEQDENSE_blocked_1(Q,R));
    //PetscCall(MatMatMultQRInplace_SEQDENSE_blocked_2(Q,R));
  } else {
    PetscCall(MatMatMultQRInplace_MPIDENSE(Q,R));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultXtX_SEQDENSE(Mat X,Mat *xtx)
{
  //Mat Xt;

  PetscFunctionBeginUser;
  //PetscCall(MatTranspose(X,MAT_INITIAL_MATRIX,&Xt));
  PetscCall(MatTransposeMatMult(X, X, MAT_INITIAL_MATRIX, 1.0, xtx));
  //PetscCall(MatDestroy(&Xt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultXtX_MPIDENSE(Mat X,Mat *_xtx)
{
  Mat Xl,xtx,xtx_sum = NULL;
  PetscScalar *_xtx_array = NULL,*_xtx_sum_array = NULL;
  PetscInt M,N,count;
  MPI_Comm comm;
  PetscMPIInt commsize;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)X);
  MPI_Comm_size(comm,&commsize);
  PetscCall(MatDenseGetLocalMatrix(X,&Xl));
  PetscCall(MatMatMultXtX_SEQDENSE(Xl,&xtx));

  PetscCall(MatGetSize(xtx,&M,&N));
  count = 0;
  if (_xtx) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, M, N, NULL, &xtx_sum));
    count = 1;
  }
  if (xtx_sum) {
    PetscCall(MatDenseGetArray(xtx_sum,&_xtx_sum_array));
  }

  PetscCall(MatDenseGetArray(xtx,&_xtx_array));

  /* count how many ranks requested xtx */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, comm));

  if (count == 1) {
    PetscCallMPI(MPI_Reduce(_xtx_array, _xtx_sum_array, M*N, MPIU_SCALAR, MPIU_SUM, 0, comm));
  } else if (count == commsize){
    PetscCallMPI(MPI_Allreduce(_xtx_array, _xtx_sum_array, M*N, MPIU_SCALAR, MPIU_SUM, comm));
  } else SETERRQ(comm,PETSC_ERR_SUP,"Only supports rank=0 or global reduction of X^TX");

  PetscCall(MatDenseRestoreArray(xtx,&_xtx_array));

  if (xtx_sum) {
    PetscCall(MatDenseRestoreArray(xtx_sum,&_xtx_sum_array));
  }
  if (_xtx) {
    *_xtx = xtx_sum;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultXtX_DENSE(Mat X,Mat *_xtx)
{
  PetscMPIInt commsize;

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_DEBUG)
    {
      PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
      PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&isdense[0]);
      PetscObjectTypeCompare((PetscObject)X,MATMPIDENSE,&isdense[1]);
      if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
  #endif

  MPI_Comm_size(PetscObjectComm((PetscObject)X),&commsize);
  if (commsize == 1) {
    PetscCall(MatMatMultXtX_SEQDENSE(X,_xtx));
  } else {
    PetscCall(MatMatMultXtX_MPIDENSE(X,_xtx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// rank 0 only
PetscErrorCode test1_xtx(PetscInt M,PetscInt N)
{
  Mat X,xtx = NULL,*p2xtx = NULL;
  PetscLogDouble t0,t1;
  PetscMPIInt commsize,commrank;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  MPI_Comm_rank(PETSC_COMM_WORLD,&commrank);
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &X));
  PetscCall(MatSetRandom(X,NULL));
  if (commrank == 0) { p2xtx = &xtx; }
  PetscTime(&t0);
  PetscCall(MatMatMultXtX_DENSE(X,p2xtx));
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"[xtx] commsize %d | %1.6ld x %1.6ld -> %1.6e (s)\n",(int)commsize,(long int)M,(long int)N,(double)(t1-t0));
  PetscCall(MatDestroy(&X));
  printf("rank %d ptr %p\n",commrank,xtx);
  PetscCall(MatDestroy(&xtx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test2_xtx(PetscInt M,PetscInt N)
{
  Mat X,xtx = NULL;
  PetscLogDouble t0,t1;
  PetscMPIInt commsize,commrank;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  MPI_Comm_rank(PETSC_COMM_WORLD,&commrank);
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &X));
  PetscCall(MatSetRandom(X,NULL));
  PetscTime(&t0);
  PetscCall(MatMatMultXtX_DENSE(X,&xtx));
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"[xtx] commsize %d | %1.6ld x %1.6ld -> %1.6e (s)\n",(int)commsize,(long int)M,(long int)N,(double)(t1-t0));
  PetscCall(MatDestroy(&X));
  //MatView(xtx,PETSC_VIEWER_STDOUT_SELF);
  printf("rank %d ptr %p\n",commrank,xtx);
  PetscCall(MatDestroy(&xtx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1_QRinplace(PetscInt M,PetscInt N)
{
  Mat X0,X,xtx = NULL,xtx_red;
  PetscLogDouble t0,t1;
  PetscMPIInt commsize,commrank;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  MPI_Comm_rank(PETSC_COMM_WORLD,&commrank);
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &X));
  PetscCall(MatSetRandom(X,NULL));

  PetscCall(MatDuplicate(X,MAT_COPY_VALUES,&X0));

  MatCopy(X0,X,SAME_NONZERO_PATTERN);
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &xtx));
  PetscCall(MatSetRandom(xtx,NULL));
  {
    PetscCall(MatCreateRedundantMatrix(xtx,commsize,MPI_COMM_NULL,MAT_INITIAL_MATRIX,&xtx_red));
  }

  MatCopy(X0,X,SAME_NONZERO_PATTERN);
  PetscTime(&t0);
  PetscCall(MatMatMultQRInplace_DENSE(X,xtx_red));
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"[x.xtx inplace] commsize %d | %1.6ld x %1.6ld -> %1.6e (s)\n",(int)commsize,(long int)M,(long int)N,(double)(t1-t0));

  Mat C;
  PetscTime(&t0);
  MatMatMult(X0,xtx,MAT_INITIAL_MATRIX,1.0,&C);
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"[x.xtx inplace-direct] commsize %d | %1.6ld x %1.6ld -> %1.6e (s)\n",(int)commsize,(long int)M,(long int)N,(double)(t1-t0));

  PetscReal nrm;
  PetscCall(MatAXPY(C,-1.0,X,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C, NORM_INFINITY,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"||xxx||_infty %1.12e\n",nrm));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&X0));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&xtx));
  PetscCall(MatDestroy(&xtx_red));
  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  PetscInt M=6,N=5,test_idx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-tidx",&test_idx,NULL));
  switch (test_idx) {
    case 0:
    PetscCall(test1_xtx(M,N));
    break;

    case 1:
    PetscCall(test2_xtx(M,N));
    break;

    case 2:
    PetscCall(test1_QRinplace(M,N));
    break;
  }

  PetscCall(PetscFinalize());
  return 0;
}
