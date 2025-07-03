static char help[] = "SVD demo.";

#include <petsc.h>
#include <petscblaslapack.h>

// https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/dgesvd-example-c.html
/* Parameters */
#define M 6
#define N 5
#define LDA M
#define LDU M
#define LDVT N

/* Auxiliary routine: printing a matrix */
PetscErrorCode print_matrix( const char desc[], int m, int n, double* a, int lda ) {
        int i, j;

        PetscFunctionBeginUser;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %+1.12e [%d] ", a[i+j*lda],i+j*lda );
                printf( "\n" );
        }
        PetscFunctionReturn(PETSC_SUCCESS);
}
/* Main program */
PetscErrorCode intel_example(void)
{
        /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        /* Local arrays */
        double s[N], u[LDU*M], vt[LDVT*N];
        double a[LDA*N] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };

        PetscFunctionBeginUser;
        /* Executable statements */
        printf( " DGESVD Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        //dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info );
        PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_("A", "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info));

        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Compute SVD */
        //dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info );
        PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_("A", "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info));

        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_matrix( "Singular values", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        /* Free workspace */
        free( (void*)work );
        PetscFunctionReturn(PETSC_SUCCESS);
} /* End of DGESVD Example */

PetscErrorCode petsc_array_svd(
  const PetscScalar _a[],PetscBLASInt m, PetscBLASInt n,
  PetscScalar *_u[], PetscReal *_s[], PetscScalar *_vt[])
{
  PetscBool overwrite_a = PETSC_TRUE;
  char gesvd_u_opt = 'O';
  PetscBLASInt lda = m, ldu = m, ldvt = n, info, lwork;
  PetscReal wkopt;
  PetscReal *work = NULL;
  PetscScalar *a = (PetscScalar*)_a;
  PetscScalar *u = NULL, *vt = NULL;
  PetscReal *s = NULL;
  #if defined(PETSC_USE_COMPLEX)
  #endif

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for complex. Only valid for real matrices.");
  #endif
  if (n > m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for tall-skinny and sqaure matrices");
  if (!_u) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for u[]");
  if (!_s) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for s[]");
  if (!_vt) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for vt[]");

  if (overwrite_a == PETSC_TRUE) {
    gesvd_u_opt = 'O';
    u = NULL;
  } else {
    char gesvd_u_opt = 'A';
    if (*_u) { u = *_u;  PetscCall(PetscMemzero(u,m*n*sizeof(PetscScalar))); }
    else {     PetscCall(PetscCalloc1(m*n, &u)); }
  }

  if (*_s) { s = *_s; PetscCall(PetscMemzero(s,n*sizeof(PetscReal))); }
  else {     PetscCall(PetscCalloc1(n, &s)); }

  if (*_vt) { vt = *_vt; PetscCall(PetscMemzero(vt,n*n*sizeof(PetscScalar))); }
  else {      PetscCall(PetscCalloc1(n*n, &vt)); }

  /* Query and allocate the optimal workspace */
  lwork = -1;
  #if defined(PETSC_USE_COMPLEX)
  // complex call requires additional data and gesvd() has additional arg.
  #else
  PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(&gesvd_u_opt, "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info));
  #endif

  lwork = (PetscBLASInt)wkopt;
  PetscCall(PetscCalloc1(lwork, &work));
  #if defined(PETSC_USE_COMPLEX)
  #endif

  /* Compute SVD */
  #if defined(PETSC_USE_COMPLEX)
  // complex call requires additional data and gesvd() has additional arg.
  #else
  PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(&gesvd_u_opt, "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info));
  #endif

  /* Check for convergence */
  if (info > 0) {
    printf( "LAPACK algorithm (gesvd) for computing the SVD failed to converge.\n" );
    PetscFunctionReturn(PETSC_ERR_LIB);
  }
  if (overwrite_a == PETSC_TRUE) {
    *_u = a;
  } else {
    *_u = u;
  }
  *_s = s;
  *_vt = vt;
  PetscCall(PetscFree(work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PETSc missing prototype */
#define LAPACKgesdd_  PETSCBLAS(gesdd, GESDD)

BLAS_EXTERN void LAPACKgesdd_(const char *, const PetscBLASInt *, const PetscBLASInt *,
  PetscScalar *, const PetscBLASInt *,
  PetscReal *, PetscScalar *, const PetscBLASInt *,
  PetscScalar *, const PetscBLASInt *,
  PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

//extern void dgesdd( char* jobz, int* m, int* n, double* a,
//                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
//                double* work, int* lwork, int* iwork, int* info );

PetscErrorCode petsc_array_svd_dc(
  const PetscScalar _a[],PetscBLASInt m, PetscBLASInt n,
  PetscScalar *_u[], PetscReal *_s[], PetscScalar *_vt[])
{
  PetscBool overwrite_a = PETSC_TRUE;
  char gesvd_u_opt = 'O';
  PetscBLASInt lda = m, ldu = m, ldvt = n, info, lwork;
  PetscReal wkopt;
  PetscReal *work = NULL;
  PetscBLASInt *iwork = NULL; /* iwork dimension should be at least 8*min(m,n) */
  PetscScalar *a = (PetscScalar*)_a;
  PetscScalar *u = NULL, *vt = NULL;
  PetscReal *s = NULL;

  PetscFunctionBeginUser;
  #if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for complex. Only valid for real matrices.");
  #endif
  if (n > m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid for tall-skinny and sqaure matrices");
  if (!_u) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for u[]");
  if (!_s) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for s[]");
  if (!_vt) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Require pointer for vt[]");

  if (overwrite_a == PETSC_TRUE) {
    gesvd_u_opt = 'O';
    u = NULL;
  } else {
    char gesvd_u_opt = 'A';
    if (*_u) { u = *_u; PetscCall(PetscMemzero(u,m*n*sizeof(PetscScalar))); }
    else {     PetscCall(PetscCalloc1(m*n, &u)); }
  }

  if (*_s) { s = *_s; PetscCall(PetscMemzero(s,n*sizeof(PetscReal))); }
  else {     PetscCall(PetscCalloc1(n, &s)); }

  if (*_vt) { vt = *_vt; PetscCall(PetscMemzero(vt,n*n*sizeof(PetscScalar))); }
  else {      PetscCall(PetscCalloc1(n*n, &vt)); }

  PetscCall(PetscCalloc1(8*n, &iwork));

  /* Query and allocate the optimal workspace */
  lwork = -1;
  PetscCallBLAS("LAPACKgesdd", LAPACKgesdd_(&gesvd_u_opt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, &info));

  lwork = (PetscBLASInt)wkopt;
  PetscCall(PetscCalloc1(lwork, &work));

  /* Compute SVD */
  PetscCallBLAS("LAPACKgesdd", LAPACKgesdd_(&gesvd_u_opt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info));

  /* Check for convergence */
  if (info > 0) {
    printf( "LAPACK algorithm (gesvd) for computing the SVD failed to converge.\n" );
    PetscFunctionReturn(PETSC_ERR_LIB);
  }
  if (overwrite_a == PETSC_TRUE) {
    *_u = a;
  } else {
    *_u = u;
  }
  *_s = s;
  *_vt = vt;
  PetscCall(PetscFree(iwork));
  PetscCall(PetscFree(work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PETScSVD(Mat A, Mat *u, Vec *s, Mat *vt, PetscBool overwrite_A_with_U)
{
  PetscInt m,n;
  PetscScalar *_u_data = NULL;
  PetscScalar *_s_data = NULL;
  PetscScalar *_vt_data = NULL;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A,&m,&n));
  if (overwrite_A_with_U == PETSC_TRUE) {
    PetscScalar *_A_data = NULL;

    PetscCall(MatDenseGetArray(A,&_A_data));
    PetscCall(PetscObjectReference((PetscObject)A));
    *u = A;

    /* I want vt to be free'd when MatDestroy(&vt) is called */
    //1 PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, _vt_data, vt));
    // or
    //2 PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, vt));
    // 2 PetscCall(MatDenseReplaceArray(*vt, (const PetscScalar*)_vt_data));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, vt));
    PetscCall(MatDenseGetArray(*vt,&_vt_data));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, s));
    PetscCall(VecGetArray(*s,&_s_data));

    PetscCall(petsc_array_svd(_A_data, m, n, &_u_data, &_s_data, &_vt_data));

    PetscCall(VecRestoreArray(*s,&_s_data));
    PetscCall(MatDenseRestoreArray(*vt,&_vt_data));
    PetscCall(MatDenseRestoreArray(A,&_A_data));
  } else {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,u));
    PetscCall(MatDenseGetArray(*u,&_u_data));

    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, vt));
    PetscCall(MatDenseGetArray(*vt,&_vt_data));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, s));
    PetscCall(VecGetArray(*s,&_s_data));

    PetscCall(petsc_array_svd(_u_data, m, n, &_u_data, &_s_data, &_vt_data));

    PetscCall(VecRestoreArray(*s,&_s_data));
    PetscCall(MatDenseRestoreArray(*vt,&_vt_data));
    PetscCall(MatDenseRestoreArray(*u,&_u_data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1(void)
{
  double a[] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
  PetscScalar *u=NULL,*vt=NULL;
  PetscReal *s=NULL;

  PetscFunctionBeginUser;
  PetscCall(print_matrix( "a0", 6, 5, a, 6 ));

  PetscCall(petsc_array_svd(a, 6, 5, &u, &s, &vt));
  //PetscCall(petsc_array_svd_dc(a, 6, 5, &u, &s, &vt));

  PetscCall(print_matrix( "a1", 6, 5, a, 6 ));
  PetscCall(print_matrix( "Singular values", 1, 5, s, 1 ));
  PetscCall(print_matrix( "Left singular vectors (stored columnwise)", 6, 5, u, 6 ));
  PetscCall(print_matrix( "Right singular vectors (stored rowwise)", 5, 5, vt, 5 ));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode checksvd(Mat A, Mat u, Vec s, Mat vt)
{
  Mat svt,Asvd;
  PetscReal nrm;
  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(vt,MAT_COPY_VALUES,&svt));
  PetscCall(MatDiagonalScale(svt, s, NULL));
  PetscCall(MatMatMult(u,svt, MAT_INITIAL_MATRIX,1.0,&Asvd));
  PetscCall(MatAXPY(Asvd,-1.0,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Asvd, NORM_INFINITY,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"||A - u.diag(s).vt||_infty %1.12e\n",nrm));
  PetscCall(MatDestroy(&svt));
  PetscCall(MatDestroy(&Asvd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Do not override A */
PetscErrorCode test2(void)
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
  Mat uu,vtvt;
  Vec ss;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 6, 5, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArray(A, &data));
  PetscCall(PetscMemcpy(data,a,sizeof(PetscScalar)*6*5));
  PetscCall(MatDenseRestoreArray(A, &data));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PETScSVD(A,&uu,&ss,&vtvt,PETSC_FALSE));
  PetscCall(checksvd(A,uu,ss,vtvt));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a1\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Singular values\n"));
  PetscCall(VecView(ss,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Left singular vectors\n"));
  PetscCall(MatView(uu,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Right singular vectors\n"));
  PetscCall(MatView(vtvt,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&uu));
  PetscCall(VecDestroy(&ss));
  PetscCall(MatDestroy(&vtvt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Override A */
PetscErrorCode test3(void)
{
  double a[] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
  Mat A,B;
  PetscScalar *data;
  Mat uu,vtvt;
  Vec ss;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 6, 5, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArray(A, &data));
  PetscCall(PetscMemcpy(data,a,sizeof(PetscScalar)*6*5));
  PetscCall(MatDenseRestoreArray(A, &data));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a0\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  PetscCall(PETScSVD(A,&uu,&ss,&vtvt,PETSC_TRUE));
  PetscCall(checksvd(B,uu,ss,vtvt));
  PetscCall(MatDestroy(&B));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"a1\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Singular values\n"));
  PetscCall(VecView(ss,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Left singular vectors\n"));
  PetscCall(MatView(uu,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Right singular vectors\n"));
  PetscCall(MatView(vtvt,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&uu));
  PetscCall(VecDestroy(&ss));
  PetscCall(MatDestroy(&vtvt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test4(void)
{
  Mat A;
  PetscScalar *data;
  Mat uu,vtvt;
  Vec ss;
  PetscInt i,j,MM,NN;
  PetscLogDouble t0,t1;

  PetscFunctionBeginUser;
  MM = 6;
  NN = 3;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&MM,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&NN,NULL));
  if (NN > MM) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for square or tall skinny matrices");
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, MM, NN, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArray(A, &data));
  for (i=0; i<NN; i++) { // use min(MM,NN)
    j = i;
    data[i+j*MM] = 1.0;
  }
  PetscCall(MatDenseRestoreArray(A, &data));
  //PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  PetscTime(&t0);
  PetscCall(PETScSVD(A,&uu,&ss,&vtvt,PETSC_FALSE));
  PetscTime(&t1);
  PetscPrintf(PETSC_COMM_WORLD,"%1.6d x %1.6d %1.6e (s)\n",(int)MM,(int)NN,(double)(t1-t0));
  PetscCall(checksvd(A,uu,ss,vtvt));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&uu));
  PetscCall(VecDestroy(&ss));
  PetscCall(MatDestroy(&vtvt));
  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  //intel_example();
  //test1();
  //test2();
  //test3();
  test4();

  PetscCall(PetscFinalize());
  return 0;
}
