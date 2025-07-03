static char help[] = "X^TX demo.";

#include <petsc.h>

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
  }

  PetscCall(PetscFinalize());
  return 0;
}
