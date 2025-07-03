static char help[] = "GS, QR demo.";

#include <petsc.h>
#include <petscblaslapack.h>

/*
Create a square matrix B which can be used for the product
C = A B
*/
PetscErrorCode MatFactCreateCompat(Mat A,PetscScalar *data,Mat *B)
{
  PetscInt yM,ym,xM,xm;
  Vec x,y;

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_DEBUG)
    {
      PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]));
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]));
      if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }
  #endif
  PetscCall(MatCreateVecs(A,&x,&y));

  PetscCall(VecGetSize(y,&yM));
  PetscCall(VecGetLocalSize(y,&ym));
  PetscCall(VecGetSize(x,&xM));
  PetscCall(VecGetLocalSize(x,&xm));

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A), xm, xm, xM, xM, data, B));
  PetscCall(MatSetUp(*B));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Creates a random matrix which is indepednent of the partition size (# of MPI ranks)
 Q must be of MatType MATDENSE
 */
PetscErrorCode MatDenseSetRandomConsistent(Mat Q)
{
  Vec               q;
  PetscInt          m,M,N,j,i,start,end;
  PetscRandom       randObj;
  const PetscScalar *_q;
  PetscScalar       *_Q;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense[0]));
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATMPIDENSE,&isdense[1]));
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  PetscCall(MatGetSize(Q,&M,&N));
  PetscCall(MatGetLocalSize(Q,&m,NULL));

  PetscCall(VecCreate(PETSC_COMM_SELF,&q));
  PetscCall(VecSetSizes(q,PETSC_DECIDE,M));
  PetscCall(VecSetType(q,VECSEQ));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&randObj));
  PetscCall(PetscRandomSetFromOptions(randObj));

  PetscCall(MatGetOwnershipRange(Q,&start,&end));
  PetscCall(MatDenseGetArray(Q,&_Q));
  for (j=0; j<N; j++) {
    PetscCall(VecSetRandom(q,randObj));
    PetscCall(VecGetArrayRead(q,&_q));
    for (i=start; i<end; i++) {
      _Q[(i-start) + j*m] = _q[i];
    }
    PetscCall(VecRestoreArrayRead(q,&_q));
  }
  PetscCall(MatDenseRestoreArray(Q,&_Q));

  PetscCall(MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscRandomDestroy(&randObj));
  PetscCall(VecDestroy(&q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
Computes C = A^T A
  A is MATDENSE (SEQ or MPI)
  C is MATSEQDENSE
*/
PetscErrorCode MatTransposeMatMult_DENSE(Mat A, MatReuse scall, Mat *_C)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
Computes C = A B
  A is MATDENSE (SEQ or MPI)
  B is MATSEQDENSE
*/
PetscErrorCode MatMatMult_DENSE_SEQDENSE(Mat A, Mat B, MatReuse scall, Mat *_C)
{
  PetscInt MA,NB,mA,nB;
  Mat C,lA,lC;

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_DEBUG)
    {
      PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]));
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]));
      if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid with A using MatType MATSEQDENSE or MATMPIDENSE");

      PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&isdense[0]));
      if (!isdense[0]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid with B using MatType MATSEQDENSE");
    }
  #endif

  PetscCall(MatGetSize(A,&MA,NULL));
  PetscCall(MatGetLocalSize(A,&mA,NULL));
  PetscCall(MatGetSize(B,NULL,&NB));
  PetscCall(MatGetLocalSize(B,NULL,&nB));

  if (scall ==  MAT_REUSE_MATRIX) {
    C = *_C;
  } else if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A), mA, PETSC_DECIDE, MA, NB, NULL, &C));
    PetscCall(MatSetUp(C));
  }
  PetscCall(MatDenseGetLocalMatrix(A,&lA));
  PetscCall(MatDenseGetLocalMatrix(C,&lC));
  PetscCall(MatMatMult(lA,B,MAT_REUSE_MATRIX,1.0,&lC));

  *_C = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecOrthonormalize_ModifiedGramSchmidt(PetscInt n,const Vec X[])
{
  Vec            q,v;
  PetscInt       j,k;
  PetscReal      nrm,dot;

  PetscFunctionBeginUser;
  for (j=0; j<n; j++) {
    q = X[j];
    PetscCall(VecNorm(q,NORM_2,&nrm));
    PetscCall(VecScale(q,1.0/nrm));
    for (k=j+1; k<n; k++) {
      v = X[k];
      PetscCall(VecDot(q,v,&dot));
      PetscCall(VecAXPY(v,-dot,q));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecOrthonormalize(PetscInt n,const Vec X[])
{
  PetscFunctionBeginUser;
  PetscCall(VecOrthonormalize_ModifiedGramSchmidt(n,X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatOrthonormalize_Dense(Mat Q)
{
  PetscInt       j,M,m,N,bs;
  Vec            *q;
  PetscScalar    *array;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense[0]));
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATMPIDENSE,&isdense[1]));
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  /* Create array of vectors */
  PetscCall(MatGetSize(Q,&M,&N));
  PetscCall(MatGetLocalSize(Q,&m,NULL));
  PetscCall(MatGetBlockSize(Q,&bs));

  PetscCall(MatDenseGetArray(Q,&array));

  comm = PetscObjectComm((PetscObject)Q);
  PetscCall(PetscCalloc1(N,&q));
  for (j=0; j<N; j++) {
    PetscScalar *array_j = array + j * m;
    PetscCall(VecCreateMPIWithArray(comm,bs,m,M,(const PetscScalar*)array_j,&q[j]));
  }

  PetscCall(MatDenseRestoreArray(Q,&array));

  PetscCall(VecOrthonormalize(N,(const Vec*)q));

  for (j=0; j<N; j++) {
    PetscCall(VecDestroy(&q[j]));
  }
  PetscCall(PetscFree(q));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _VecQRMGS(PetscInt n,const Vec X[],PetscScalar *_R[])
{
  Vec            q,v;
  PetscInt       j,k;
  PetscReal      nrm,dot;
  PetscScalar    *R = NULL;

  PetscFunctionBeginUser;
  if (!_R) SETERRQ(PetscObjectComm((PetscObject)X[0]),PETSC_ERR_SUP,"R must be non-NULL");
  if (*_R) {
    R = *_R;
    PetscCall(PetscMemzero(R,n*n*sizeof(PetscScalar)));
  } else {
    PetscCall(PetscCalloc1(n*n,&R));
  }
  for (j=0; j<n; j++) {
    if (j%100 == 0) PetscPrintf(PETSC_COMM_WORLD,"  progress %d / %d\n",j,n);
    q = X[j];
    PetscCall(VecNorm(q,NORM_2,&nrm));
    PetscCall(VecScale(q,1.0/nrm));
    R[j + j*n] = nrm;
    for (k=j+1; k<n; k++) {
      v = X[k];
      PetscCall(VecDot(q,v,&dot));
      PetscCall(VecAXPY(v,-dot,q));
      R[j + k*n] = dot;
    }
  }
  *_R = R;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatQRMGS_Dense(Mat Q,Mat R)
{
  PetscInt       j,M,m,N,bs;
  Vec            *q;
  PetscScalar    *array,*array_R;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense[0]));
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATMPIDENSE,&isdense[1]));
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  /* Create array of vectors */
  PetscCall(MatGetSize(Q,&M,&N));
  PetscCall(MatGetLocalSize(Q,&m,NULL));
  PetscCall(MatGetBlockSize(Q,&bs));

  PetscCall(MatDenseGetArray(Q,&array));

  comm = PetscObjectComm((PetscObject)Q);
  PetscCall(PetscCalloc1(N,&q));
  for (j=0; j<N; j++) {
    PetscScalar *array_j = array + j * m;
    PetscCall(VecCreateMPIWithArray(comm,bs,m,M,(const PetscScalar*)array_j,&q[j]));
  }

  PetscCall(MatDenseRestoreArray(Q,&array));

  PetscCall(MatDenseGetArray(R,&array_R));
  PetscCall(_VecQRMGS(N,(const Vec*)q,&array_R));
  PetscCall(MatDenseRestoreArray(R,&array_R));

  for (j=0; j<N; j++) {
    PetscCall(VecDestroy(&q[j]));
  }
  PetscCall(PetscFree(q));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatQRMGS(Mat A,Mat *Q,Mat *R)
{
  PetscInt M,N;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, N, N, NULL, R));
  PetscCall(_MatQRMGS_Dense(A,*R));
  PetscObjectReference((PetscObject)A);
  *Q = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatDenseCreateEmptyColumnVec(Mat A,Vec *v)
{
  PetscInt       M,N,m,n,bs;
  PetscMPIInt    commsize;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]);
    PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]);
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetBlockSize(A,&bs));
  MPI_Comm_size(PetscObjectComm((PetscObject)A),&commsize);
  if (commsize == 1) {
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,M,NULL,v));
  } else {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),bs,m,M,NULL,v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatDenseSetColumnVec(Mat A,PetscInt j,Vec v)
{
  PetscInt       lda;
  PetscScalar    *array;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]);
    PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]);
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  PetscCall(MatDenseGetLDA(A,&lda));
  PetscCall(MatDenseGetArray(A,&array));
  PetscCall(VecPlaceArray(v,array + (size_t)j * (size_t)lda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatDenseResetColumnVec(Mat A,PetscInt j,Vec v)
{
  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]);
    PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]);
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif
  PetscCall(VecResetArray(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPull(Mat Qhat,Mat X,Mat *Xhat)
{
  Mat Xhatlayout,*submat;
  PetscInt s,e,mlocal,QN;
  IS isrow,iscol;
  MPI_Comm       comm;
  PetscMPIInt    commrank;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)Qhat);
  MPI_Comm_rank(comm,&commrank);
  PetscCall(MatGetSize(Qhat,NULL,&QN));
  PetscCall(MatFactCreateCompat(Qhat, NULL, &Xhatlayout));

  /*
  {
    PetscInt MM,NN,mm,nn,MMl,NNl;
    Mat Xl;

    MatGetSize(Xhatlayout,&MM,&NN);
    MatGetLocalSize(Xhatlayout,&mm,&nn);
    MatDenseGetLocalMatrix(Xhatlayout,&Xl);
    MatGetSize(Xl,&MMl,&NNl);
    PetscPrintf(PETSC_COMM_SELF,"[first] rank %d - %d x %d - %d x %d --> local mat %d x %d\n",commrank,MM,NN,mm,nn,MMl,NNl);
  }
  */

  PetscCall(MatGetOwnershipRange(Xhatlayout,&s,&e));
  mlocal = e - s;

  PetscCall(ISCreateStride(PETSC_COMM_SELF, mlocal, s, 1, &isrow));
  if (mlocal == 0) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &iscol));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, QN, 0, 1, &iscol));
  }

  PetscCall(MatCreateSubMatrices(X, 1, (const IS*)&isrow, (const IS*)&iscol, MAT_INITIAL_MATRIX, &submat));

  /*
  {
    PetscInt MM,NN,mm,nn;

    if (mlocal == 0) {
      PetscPrintf(PETSC_COMM_SELF,"[second] rank %d - empty --\n",commrank);
    } else {
      MatGetSize(submat[0],&MM,&NN);
      MatGetLocalSize(submat[0],&mm,&nn);
      PetscPrintf(PETSC_COMM_SELF,"[second] rank %d - %d x %d - %d x %d\n",commrank,MM,NN,mm,nn);
    }
  }
  */

  {
    if (mlocal == 0) {
    } else {
      Mat Xl;

      PetscCall(MatDenseGetLocalMatrix(Xhatlayout,&Xl));
      PetscCall(MatCopy(submat[0],Xl,SAME_NONZERO_PATTERN));
    }
  }

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  if (submat) {
    if (submat[0]) {
      PetscCall(MatDestroy(&submat[0]));
    }
    PetscCall(PetscFree(submat));
  }

  *Xhat = Xhatlayout;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatQRBlockMGS_Dense_v2_SEQ(Mat Q,Mat R)
{
  PetscInt       k;
  Mat            T,Qhat,That,Rhat;
  MPI_Comm       comm;
  PetscInt       QM,QN,Qm,Qn;
  Vec            xk,qk,yk;
  PetscScalar    *_Q;
  PetscReal      rkk;
  PetscMPIInt    commrank;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense = {PETSC_FALSE};
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense));
    if (!isdense) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE");
  }
#endif

  comm = PetscObjectComm((PetscObject)Q);
  PetscCall(MatDuplicate(R,MAT_DO_NOT_COPY_VALUES,&T));

  PetscCall(MatGetSize(Q,&QM,&QN));
  PetscCall(MatGetLocalSize(Q,&Qm,&Qn));

  PetscCall(MatDenseGetArray(Q,&_Q));
  PetscCall(MatCreateDense(comm, Qm, PETSC_DECIDE, QM, 1, _Q, &Qhat));
  PetscCall(MatDenseRestoreArray(Q,&_Q));

  //
  PetscCall(_MatDenseCreateEmptyColumnVec(Q,&xk));
  PetscCall(MatCreateVecs(Q,NULL,&qk));
  PetscCall(VecDuplicate(qk,&yk));

  PetscCall(_MatDenseSetColumnVec(Q,0,xk));

  PetscCall(VecNorm(xk,NORM_2,&rkk)); //rkk = np.linalg.norm(xk)
  PetscCall(VecCopy(xk,qk));
  PetscCall(VecScale(qk,1.0/rkk)); // qk = np.copy(xk) / rkk

  // Q[:, 0] = qk[:]
  PetscCall(VecCopy(qk,xk));
  PetscCall(_MatDenseResetColumnVec(Q,0,xk));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &Rhat));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &That));

  // R[0, 0] = rkk
  // T[0, 0] = 1.0
  {
    PetscScalar *_R,*_T,*_Rhat,*_That;
    PetscInt RM = QN, hatM = 1;

    PetscCall(MatDenseGetArray(R,&_R));
    PetscCall(MatDenseGetArray(Rhat,&_Rhat));

    PetscCall(MatDenseGetArray(T,&_T));
    PetscCall(MatDenseGetArray(That,&_That));

    _R[0 + 0*RM] = rkk;
    _Rhat[0 + 0*hatM] = _R[0 + 0 * RM];

    _T[0 + 0*RM] = 1.0;
    _That[0 + 0*hatM] = _T[0 + 0 * RM];

    PetscCall(MatDenseRestoreArray(That,&_That));
    PetscCall(MatDenseRestoreArray(T,&_T));

    PetscCall(MatDenseRestoreArray(Rhat,&_Rhat));
    PetscCall(MatDenseRestoreArray(R,&_R));
  }
  //
  PetscPrintf(comm,"That_");
  MatView(That,PETSC_VIEWER_STDOUT_(comm));

  PetscPrintf(comm,"Rhat_");
  MatView(Rhat,PETSC_VIEWER_STDOUT_(comm));

  for (k=1; k<QN; k++) {
    Vec hk,hk2;

    PetscCall(MatCreateVecs(Qhat,&hk,NULL));
    PetscCall(VecDuplicate(hk,&hk2));

    if (k%100 == 0) PetscPrintf(PETSC_COMM_WORLD,"  progress %d / %d\n",k,QN);

    //xk = Q[:, k] # This is really X
    PetscCall(_MatDenseSetColumnVec(Q,k,xk));

    // hk = np.matmul(Qhat.T, xk)
    PetscCall(MatMultTranspose(Qhat,xk,hk));

    // hk = That.T @ hk
    PetscCall(MatMultTranspose(That,hk,hk2));

    // yk = xk - Qhat @ hk
    PetscCall(MatMult(Qhat,hk2,yk));
    PetscCall(VecAYPX(yk,-1.0,xk));

    // rkk = np.linalg.norm(yk)
    PetscCall(VecNorm(yk,NORM_2,&rkk)); //rkk = np.linalg.norm(yk)
    PetscCall(VecCopy(yk,qk));
    PetscCall(VecScale(qk,1.0/rkk)); //qk = np.copy(yk) / rkk

    // insert new values into Q
    PetscCall(VecCopy(qk, xk));

    PetscCall(_MatDenseResetColumnVec(Q,k,xk));

    // insert new values into R
    // R[0:nvals, k] = hk[:]
    // R[k, k] = rkk
    {
      PetscInt ii,ss,ee;
      const PetscScalar *_h;
      PetscScalar *_R;
      PetscInt RM = QN;

      PetscCall(MatDenseGetArray(R,&_R));
      PetscCall(VecGetArrayRead(hk2,&_h));
      PetscCall(VecGetOwnershipRange(hk2,&ss,&ee));
      for (ii=0; ii<ee-ss; ii++) {
        _R[ii + k * RM] = _h[ii];
      }
      _R[k + RM * k] = rkk;
      PetscCall(VecRestoreArrayRead(hk2,&_h));
      PetscCall(MatDenseRestoreArray(R,&_R));
    }

    // gk = np.matmul(Qhat.T, qk)
    // gk = -That @ gk
    PetscCall(MatMultTranspose(Qhat,qk,hk));
    PetscCall(MatMultTranspose(That,hk,hk2));
    PetscCall(VecScale(hk2,-1.0));

    // insert new values into T
    // T[0:nvals, k] = gk[:]
    // T[k, k] = 1.0
    {
      PetscInt ii,ss,ee;
      const PetscScalar *_h;
      PetscScalar *_T;
      PetscInt TM = QN;

      PetscCall(MatDenseGetArray(T,&_T));
      PetscCall(VecGetArrayRead(hk2,&_h));
      PetscCall(VecGetOwnershipRange(hk2,&ss,&ee));
      for (ii=0; ii<ee-ss; ii++) {
        _T[ii + k * TM] = _h[ii];
      }
      _T[k + TM * k] = 1.0;
      PetscCall(VecRestoreArrayRead(hk2,&_h));
      PetscCall(MatDenseRestoreArray(T,&_T));
    }

    PetscCall(VecDestroy(&hk));
    PetscCall(VecDestroy(&hk2));

    if (k == QN-1) { break; }

    // update views
    PetscCall(MatDestroy(&Qhat));
    PetscCall(MatDestroy(&Rhat));
    PetscCall(MatDestroy(&That));

    PetscCall(MatDenseGetArray(Q,&_Q));
    PetscCall(MatCreateDense(comm, Qm, PETSC_DECIDE, QM, k+1, _Q, &Qhat));
    PetscCall(MatDenseRestoreArray(Q,&_Q));

    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, k+1, k+1, NULL, &Rhat));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, k+1, k+1, NULL, &That));
    {
      PetscScalar *_R,*_T,*_Rhat,*_That;
      PetscInt XM = QN, hatM = k+1,ii,jj;

      PetscCall(MatDenseGetArray(R,&_R));
      PetscCall(MatDenseGetArray(Rhat,&_Rhat));

      PetscCall(MatDenseGetArray(T,&_T));
      PetscCall(MatDenseGetArray(That,&_That));

      for (jj=0; jj<k+1; jj++) {
        for (ii=0; ii<k+1; ii++) {
          _Rhat[ii + jj * hatM] = _R[ii + jj * XM];
          _That[ii + jj * hatM] = _T[ii + jj * XM];
        }
      }

      PetscCall(MatDenseRestoreArray(That,&_That));
      PetscCall(MatDenseRestoreArray(T,&_T));

      PetscCall(MatDenseRestoreArray(Rhat,&_Rhat));
      PetscCall(MatDenseRestoreArray(R,&_R));
    }

  }

  PetscCall(MatDestroy(&Rhat));
  PetscCall(MatDestroy(&That));

  PetscCall(VecDestroy(&xk));
  PetscCall(VecDestroy(&qk));
  PetscCall(VecDestroy(&yk));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&Qhat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _MatQRBlockMGS_Dense_v2(Mat Q,Mat R)
{
  PetscInt       k;
  Mat            T,Qhat,That,Rhat;
  MPI_Comm       comm;
  PetscInt       QM,QN,Qm,Qn;
  Vec            xk,qk,yk;
  PetscScalar    *_Q;
  PetscReal      rkk;
  IS             isrow,iscol;
  PetscInt       i,j,s,e,mlocal,*rowidx,*colidx;
  PetscMPIInt    commrank;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&isdense[0]));
    PetscCall(PetscObjectTypeCompare((PetscObject)Q,MATMPIDENSE,&isdense[1]));
    if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)Q),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
  }
#endif

  comm = PetscObjectComm((PetscObject)Q);
  MPI_Comm_rank(comm,&commrank);
  PetscCall(MatDuplicate(R,MAT_DO_NOT_COPY_VALUES,&T));

  PetscCall(MatGetSize(Q,&QM,&QN));
  PetscCall(MatGetLocalSize(Q,&Qm,&Qn));

  PetscCall(MatDenseGetArray(Q,&_Q));
  PetscCall(MatCreateDense(comm, Qm, PETSC_DECIDE, QM, 1, _Q, &Qhat));
  PetscCall(MatDenseRestoreArray(Q,&_Q));

  //
  PetscCall(_MatDenseCreateEmptyColumnVec(Q,&xk));
  PetscCall(MatCreateVecs(Q,NULL,&qk));
  PetscCall(VecDuplicate(qk,&yk));

  PetscCall(_MatDenseSetColumnVec(Q,0,xk));

  // That, Rhat view
  //PetscCall(MatFactCreateCompat(Qhat, NULL, &That));
  //PetscCall(MatFactCreateCompat(Qhat, NULL, &Rhat));

  PetscCall(VecNorm(xk,NORM_2,&rkk)); //rkk = np.linalg.norm(xk)
  PetscCall(VecCopy(xk,qk));
  PetscCall(VecScale(qk,1.0/rkk)); // qk = np.copy(xk) / rkk

  // Q[:, 0] = qk[:]
  PetscCall(VecCopy(qk,xk));
  PetscCall(_MatDenseResetColumnVec(Q,0,xk));

  PetscCall(MatSetValue(R,0,0,rkk,INSERT_VALUES)); // R[0, 0] = rkk
  PetscCall(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetValue(T,0,0,1.0,INSERT_VALUES)); // T[0, 0] = 1.0
  PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
  //

  // That
  PetscCall(MatFactCreateCompat(Qhat, NULL, &That));

  // re-use rowidx, colidx for all iterations
  mlocal = QN; // fix-me - this is an over-estimate
  PetscCall(PetscCalloc1(mlocal,&rowidx));
  PetscCall(PetscCalloc1(QN,&colidx));

  PetscCall(MatGetOwnershipRange(That,&s,&e));
  mlocal = e - s;
  for (i=0; i<mlocal; i++) {
    rowidx[i] = s + i;
  }
  for (j=0; j<QN; j++) {
    colidx[j] = j;
  }

  PetscCall(ISCreateGeneral(comm, mlocal, (const PetscInt*)rowidx, PETSC_USE_POINTER, &isrow));
  if (mlocal == 0) {
    PetscCall(ISCreateGeneral(comm, 0, (const PetscInt*)colidx, PETSC_USE_POINTER, &iscol));
  } else {
    PetscCall(ISCreateGeneral(comm, 1, (const PetscInt*)colidx, PETSC_USE_POINTER, &iscol));
  }

  PetscCall(MatCreateSubMatrix(T, isrow, iscol, MAT_REUSE_MATRIX, &That));

  // Rhat
  PetscCall(MatFactCreateCompat(Qhat, NULL, &Rhat));
  PetscCall(MatCreateSubMatrix(R, isrow, iscol, MAT_REUSE_MATRIX, &Rhat));

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));

  PetscPrintf(comm,"That_");
  MatView(That,PETSC_VIEWER_STDOUT_(comm));

  PetscPrintf(comm,"Rhat_");
  MatView(Rhat,PETSC_VIEWER_STDOUT_(comm));

  for (k=1; k<QN; k++) {
    Vec hk,hk2;

    PetscCall(MatCreateVecs(Qhat,&hk,NULL));
    PetscCall(VecDuplicate(hk,&hk2));
    //PetscCall(MatCreateVecs(That,&hk2,NULL));

    if (k%100 == 0) PetscPrintf(PETSC_COMM_WORLD,"  progress %d / %d\n",k,QN);

    //xk = Q[:, k] # This is really X
    PetscCall(_MatDenseSetColumnVec(Q,k,xk));

    // hk = np.matmul(Qhat.T, xk)
    PetscCall(MatMultTranspose(Qhat,xk,hk));

    // hk = That.T @ hk
    PetscCall(MatMultTranspose(That,hk,hk2));

    // yk = xk - Qhat @ hk
    PetscCall(MatMult(Qhat,hk2,yk));
    PetscCall(VecAYPX(yk,-1.0,xk));

    // rkk = np.linalg.norm(yk)
    PetscCall(VecNorm(yk,NORM_2,&rkk)); //rkk = np.linalg.norm(yk)
    PetscCall(VecCopy(yk,qk));
    PetscCall(VecScale(qk,1.0/rkk)); //qk = np.copy(yk) / rkk

    // insert new values into Q
    PetscCall(VecCopy(qk, xk));

    PetscCall(_MatDenseResetColumnVec(Q,k,xk));

    // insert new values into R
    //R[0:nvals, k] = hk[:]
    {
      PetscInt ii,ss,ee;
      const PetscScalar *_h;

      PetscCall(VecGetOwnershipRange(hk2,&ss,&ee));
      PetscCall(VecGetArrayRead(hk2,&_h));
      for (ii=0; ii<ee-ss; ii++) {
        PetscCall(MatSetValue(R,ii+ss,k,_h[ii],INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(hk2,&_h));
    }
    //R[k, k] = rkk
    PetscCall(MatSetValue(R,k,k,rkk,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));

    // gk = np.matmul(Qhat.T, qk)
    // gk = -That @ gk
    PetscCall(MatMultTranspose(Qhat,qk,hk));
    PetscCall(MatMultTranspose(That,hk,hk2));
    PetscCall(VecScale(hk2,-1.0));

    // insert new values into T
    //T[0:nvals, k] = gk[:]
    {
      PetscInt ii,ss,ee;
      const PetscScalar *_h;

      PetscCall(VecGetOwnershipRange(hk2,&ss,&ee));
      PetscCall(VecGetArrayRead(hk2,&_h));
      for (ii=0; ii<ee-ss; ii++) {
        PetscCall(MatSetValue(T,ii+ss,k,_h[ii],INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(hk2,&_h));
    }
    //T[k, k] = 1.0
    PetscCall(MatSetValue(T,k,k,1.0,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

    PetscCall(VecDestroy(&hk));
    PetscCall(VecDestroy(&hk2));

    if (k == QN-1) { break; }

    // update views
    PetscCall(MatDestroy(&Qhat));
    PetscCall(MatDestroy(&Rhat));
    PetscCall(MatDestroy(&That));

    PetscCall(MatDenseGetArray(Q,&_Q));
    PetscCall(MatCreateDense(comm, Qm, PETSC_DECIDE, QM, k+1, _Q, &Qhat));
    PetscCall(MatDenseRestoreArray(Q,&_Q));

    PetscCall(MatPull(Qhat,R,&Rhat));
    PetscCall(MatPull(Qhat,T,&That));

    #if 0
    PetscCall(MatFactCreateCompat(Qhat, NULL, &That));
    PetscCall(MatFactCreateCompat(Qhat, NULL, &Rhat));
    {
      PetscInt MM,NN,mm,nn;

      PetscCall(MatGetSize(That,&MM,&NN));
      PetscCall(MatGetLocalSize(That,&mm,&nn));
      PetscPrintf(PETSC_COMM_SELF,"[first] rank %d - %d x %d - %d x %d\n",commrank,MM,NN,mm,nn);
    }

    PetscCall(MatGetOwnershipRange(That,&s,&e));
    mlocal = e - s;
    for (i=0; i<mlocal; i++) {
      rowidx[i] = s + i;
    }
    PetscCall(ISCreateGeneral(comm, mlocal, (const PetscInt*)rowidx, PETSC_USE_POINTER, &isrow));
    if (mlocal == 0) {
      PetscCall(ISCreateGeneral(comm, 0, (const PetscInt*)colidx, PETSC_USE_POINTER, &iscol));
    } else {
      PetscCall(ISCreateGeneral(comm, k+1, (const PetscInt*)colidx, PETSC_USE_POINTER, &iscol));
    }
    //ISView(iscol,PETSC_VIEWER_STDOUT_(comm));
    PetscCall(MatCreateSubMatrix(T, isrow, iscol, MAT_REUSE_MATRIX, &That));
    PetscCall(MatCreateSubMatrix(R, isrow, iscol, MAT_REUSE_MATRIX, &Rhat));
    {
      PetscInt MM,NN,mm,nn;

      PetscCall(MatGetSize(That,&MM,&NN));
      PetscCall(MatGetLocalSize(That,&mm,&nn));
      PetscPrintf(PETSC_COMM_SELF,"[second] rank %d - %d x %d - %d x %d\n",commrank,MM,NN,mm,nn);
    }

    PetscCall(ISDestroy(&isrow));
    PetscCall(ISDestroy(&iscol));

    PetscPrintf(comm,"Rhat_");
    MatView(Rhat,PETSC_VIEWER_STDOUT_(comm));

    #endif
  }

  PetscCall(MatDestroy(&Rhat));
  PetscCall(PetscFree(rowidx));
  PetscCall(PetscFree(colidx));
  PetscCall(MatDestroy(&That));

  PetscCall(VecDestroy(&xk));
  PetscCall(VecDestroy(&qk));
  PetscCall(VecDestroy(&yk));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&Qhat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatQRBlockMGS(Mat A,Mat *Q,Mat *R)
{
  PetscBool isdense[] = {PETSC_FALSE,PETSC_FALSE};

  PetscFunctionBeginUser;
    {
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isdense[0]));
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isdense[1]));
      if (!isdense[0] && !isdense[1]) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only valid for MatType MATSEQDENSE or MATMPIDENSE");
    }

  PetscCall(MatFactCreateCompat(A, NULL, R));
  if (isdense[0]) {
    PetscPrintf(PETSC_COMM_WORLD,"MatQRBlockMGS-->SEQ\n");
    PetscCall(_MatQRBlockMGS_Dense_v2_SEQ(A,*R));
  } else if (isdense[1]) {
    PetscPrintf(PETSC_COMM_WORLD,"MatQRBlockMGS-->SEQ/MPI\n");
    PetscCall(_MatQRBlockMGS_Dense_v2(A,*R));
  }

  PetscObjectReference((PetscObject)A);
  *Q = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode checkortho(Mat Q)
{
  Mat Qt,QtQ,Identity;
  PetscReal nrm;

  PetscFunctionBeginUser;
  PetscCall(MatTranspose(Q,MAT_INITIAL_MATRIX,&Qt));
  PetscCall(MatMatMult(Qt,Q, MAT_INITIAL_MATRIX,1.0,&QtQ));
  PetscCall(MatDuplicate(QtQ,MAT_DO_NOT_COPY_VALUES,&Identity));
  PetscCall(MatShift(Identity,1.0));
  PetscCall(MatAXPY(QtQ,-1.0,Identity,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(QtQ, NORM_INFINITY,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"||QtQ - I||_infty %1.12e\n",nrm));
  PetscCall(MatDestroy(&Qt));
  PetscCall(MatDestroy(&QtQ));
  PetscCall(MatDestroy(&Identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode checkqr_seq(Mat A,Mat Q,Mat R)
{
  Mat QR;
  PetscReal nrm;

  PetscFunctionBeginUser;
  PetscCall(MatMatMult(Q,R,MAT_INITIAL_MATRIX,1.0,&QR));
  PetscCall(MatAXPY(QR,-1.0,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(QR, NORM_INFINITY,&nrm));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),"||QR - A||_infty %1.12e\n",nrm));
  PetscCall(MatDestroy(&QR));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode checkqr_mpi(Mat A,Mat Q,Mat R)
{
  Mat QR;
  PetscReal nrm;

  PetscFunctionBeginUser;
  PetscCall(MatMatMult_DENSE_SEQDENSE(Q,R,MAT_INITIAL_MATRIX,&QR));
  PetscCall(MatAXPY(QR,-1.0,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(QR, NORM_INFINITY,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"||QR - A||_infty %1.12e\n",nrm));
  PetscCall(MatDestroy(&QR));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode test1_gs_seq(void)
{
  double a[] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
  Mat A;
  PetscScalar *data;
  PetscMPIInt commsize;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  if (commsize != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Uni-processor example");
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 6, 5, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArray(A, &data));
  PetscCall(PetscMemcpy(data,a,sizeof(PetscScalar)*6*5));
  PetscCall(MatDenseRestoreArray(A, &data));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatOrthonormalize_Dense(A));
  PetscCall(checkortho(A));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a1\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));


  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1_qr_seq(void)
{
  double a[] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
  Mat A,B,Q,R;
  PetscScalar *data;
  PetscMPIInt commsize;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  if (commsize != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Uni-processor example");
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 6, 5, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArray(A, &data));
  PetscCall(PetscMemcpy(data,a,sizeof(PetscScalar)*6*5));
  PetscCall(MatDenseRestoreArray(A, &data));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  PetscCall(MatQRMGS(A,&Q,&R));
  PetscCall(checkqr_seq(B,Q,R));
  PetscCall(MatDestroy(&B));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Q\n"));
  PetscCall(MatView(Q,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"R\n"));
  PetscCall(MatView(R,PETSC_VIEWER_STDOUT_SELF));


  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&R));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1_qr_mpi(void)
{
  Mat A,B,Q,R;
  PetscInt i,j,M=6,N=5,s,e;
  PetscLogDouble t0,t1;
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscMPIInt commrank;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  if (N > M) SETERRQ(comm,PETSC_ERR_SUP,"Only for square or tall skinny matrices");

  MPI_Comm_rank(comm,&commrank);
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&s,&e));
  for (i=s; i<e; i++) {
    j = i;
    if (i < N) {
      PetscCall(MatSetValue(A,i,j,(PetscScalar)(i+1),INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY));
  if (s == 0) {
    i = s;
    for (j=0; j<N; j++) {
      PetscCall(MatSetValue(A,i,j,(PetscScalar)(j+1)*0.001,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetRandom(A,NULL));
  //PetscCall(MatDenseSetRandomConsistent(A));

  //PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  //PetscCall(MatView(A,PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  PetscTime(&t0);
  PetscCall(MatQRMGS(A,&Q,&R));
  PetscTime(&t1);
  PetscPrintf(comm,"[test1_qr_mpi] %1.6ld x %1.6ld %1.6e (s)\n",(long int)M,(long int)N,(double)(t1-t0));
  PetscCall(checkqr_mpi(B,Q,R));
  PetscCall(MatDestroy(&B));

  /*
  PetscCall(PetscPrintf(comm,"Q\n"));
  PetscCall(MatView(Q,PETSC_VIEWER_STDOUT_(comm)));
  if (commrank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"R\n"));
    PetscCall(MatView(R,PETSC_VIEWER_STDOUT_SELF));
  }
  */

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&R));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1_qrblock_mpi(void)
{
  Mat A,B,Q,R;
  PetscInt i,j,M=6,N=5,s,e;
  PetscLogDouble t0,t1;
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscMPIInt commrank;

  PetscFunctionBeginUser;
  PetscPrintf(PETSC_COMM_WORLD,"%s\n",__func__);
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  if (N > M) SETERRQ(comm,PETSC_ERR_SUP,"Only for square or tall skinny matrices");

  MPI_Comm_rank(comm,&commrank);
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&s,&e));
  for (i=s; i<e; i++) {
    j = i;
    if (i < N) {
      PetscCall(MatSetValue(A,i,j,(PetscScalar)(i+1),INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY));
  if (s == 0) {
    i = s;
    for (j=0; j<N; j++) {
      PetscCall(MatSetValue(A,i,j,(PetscScalar)(j+1)*0.001,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetRandom(A,NULL));
  //PetscCall(MatDenseSetRandomConsistent(A));

  //PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  //PetscCall(MatView(A,PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  PetscTime(&t0);
  PetscCall(MatQRBlockMGS(A,&Q,&R));
  PetscTime(&t1);
  PetscPrintf(comm,"[test1_qrblock_mpi] %1.6ld x %1.6ld %1.6e (s)\n",(long int)M,(long int)N,(double)(t1-t0));

  PetscCall(checkqr_seq(B,Q,R));
  PetscCall(MatDestroy(&B));

  /*
  PetscCall(PetscPrintf(comm,"Q\n"));
  PetscCall(MatView(Q,PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(PetscPrintf(comm,"R\n"));
  PetscCall(MatView(R,PETSC_VIEWER_STDOUT_(comm)));
  */

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&R));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  PetscInt test_idx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-tidx",&test_idx,NULL));
  switch (test_idx) {
    case 0:
    PetscCall(test1_gs_seq());
    break;

    case 1:
    PetscCall(test1_qr_seq());
    break;

    case 2:
    PetscCall(test1_qr_mpi()); // time it
    break;

    case 3:
    PetscCall(test1_qrblock_mpi()); // time it
    break;
  }

  PetscCall(PetscFinalize());
  return 0;
}

// -n 4 ./ex_ortho -m 60000 -n 2000 -log_view
//  -> test1_qr_mpi -> 41 sec
//  -> test1_qrblock_mpi -> 36 sec
