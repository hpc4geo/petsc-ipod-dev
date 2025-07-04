
static char help[] = "Load a snapshot matrix demo.";

#include <petsc.h>
#include <petscviewerhdf5.h>

PetscErrorCode MatDenseCreateSnapshotFromH5(
  MPI_Comm comm,
  const char filename[],
  const char snapshot_prefix[],
  const char attr_dim_name[],
  Mat *_dataX)
{
  char vecname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  Vec x;
  Mat B;
  PetscInt i,j,M=0,N=0,mlocal=0;
  PetscScalar *_B;
  PetscMPIInt commsize;
  PetscBool has_attr_dim = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5Open(comm,filename,FILE_MODE_READ,&viewer));

  PetscCall(PetscViewerHDF5HasAttribute(viewer, "/", attr_dim_name, &has_attr_dim));
  if (has_attr_dim) {
    PetscInt val[] = {0,0};
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", attr_dim_name, PETSC_INT, NULL, val));
    PetscPrintf(comm,"Snapshot matrix dimensions %ld x %ld\n",(long int)val[0],(long int)val[1]);

    M = val[0];
    N = val[1];
  }

  j = 0;
  MPI_Comm_size(comm,&commsize);
  PetscCall(VecCreate(comm,&x));
  PetscSNPrintf(vecname,PETSC_MAX_PATH_LEN-1,"%s_%d",snapshot_prefix,0);
  PetscCall(PetscObjectSetName((PetscObject)x,vecname));
  PetscCall(VecLoad(x,viewer));

  // N from attribute or IS
  PetscCall(VecGetSize(x,&M));
  PetscCall(VecGetLocalSize(x,&mlocal));
  PetscCall(MatCreateDense(comm, mlocal, PETSC_DECIDE, M, N, NULL, &B));

  PetscCall(MatDenseGetArray(B,&_B));

  {
    PetscScalar *_vals;

    j = 0;
    PetscCall(VecGetArray(x,&_vals));
    for (i=0; i<mlocal; i++) {
      _B[i + j*N] = _vals[i];
    }
    PetscCall(VecRestoreArray(x,&_vals));
  }

  for (j=1; j<N; j++) {
    PetscScalar *_vals;

    PetscSNPrintf(vecname,PETSC_MAX_PATH_LEN-1,"%s_%d",snapshot_prefix,(int)j);
    PetscCall(PetscObjectSetName((PetscObject)x,vecname));
    PetscCall(VecLoad(x,viewer));
    //PetscCall(VecView(x,comm));

    PetscCall(VecGetArray(x,&_vals));
    for (i=0; i<mlocal; i++) {
      _B[i + j*mlocal] = _vals[i];
    }
    PetscCall(VecRestoreArray(x,&_vals));
  }
  PetscCall(MatDenseRestoreArray(B,&_B));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscViewerDestroy(&viewer));
  *_dataX = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test1(void)
{
  const char attr_dim_n[] = "attr_dim";
  const char dim_n[] = "dim";

  char prefix[] = "x";
  char fname[] = "tiny.hdf5";
  char vecname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  Vec X;
  Mat B;
  PetscInt i,j,M=0,N=0,mlocal=0;
  PetscScalar *_B;
  PetscMPIInt commsize;

  PetscFunctionBeginUser;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer));

  PetscBool has_attr_dim = PETSC_FALSE;
  PetscViewerHDF5HasAttribute(viewer, "/", attr_dim_n, &has_attr_dim);
  if (has_attr_dim) {
    PetscInt val[] = {0,0};
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", attr_dim_n, PETSC_INT, NULL,val));
    printf("found %d %d\n",val[0],val[1]);

    M = val[0];
    N = val[1];
  }

  /* Cannot load this sequentially if opened file with comm_world */
  PetscBool has_dim = PETSC_FALSE;
  PetscCall(PetscViewerHDF5HasDataset(viewer, dim_n, &has_dim));
  if (has_dim) {
    IS is;

    PetscCall(ISCreate(PETSC_COMM_WORLD,&is));
    ISSetType(is,ISGENERAL);
    PetscCall(PetscObjectSetName((PetscObject)is,"dim"));
    PetscCall(ISLoad(is,viewer));
    PetscCall(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  }

  j = 0;
  MPI_Comm_size(PETSC_COMM_WORLD,&commsize);
  PetscCall(VecCreate(PETSC_COMM_WORLD,&X));
  PetscSNPrintf(vecname,PETSC_MAX_PATH_LEN-1,"%s_%d",prefix,0);
  PetscCall(PetscObjectSetName((PetscObject)X,vecname));
  PetscCall(VecLoad(X,viewer));

  // N from attribute or IS
  PetscCall(VecGetSize(X,&M));
  PetscCall(VecGetLocalSize(X,&mlocal));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, mlocal, PETSC_DECIDE, M, N, NULL, &B));

  PetscCall(MatDenseGetArray(B,&_B));

  {
    PetscScalar *_vals;

    j = 0;
    PetscCall(VecGetArray(X,&_vals));
    for (i=0; i<mlocal; i++) {
      _B[i + j*N] = _vals[i];
    }
    PetscCall(VecRestoreArray(X,&_vals));
  }

  for (j=1; j<N; j++) {
    PetscScalar *_vals;

    PetscSNPrintf(vecname,PETSC_MAX_PATH_LEN-1,"%s_%d",prefix,(int)j);
    PetscCall(PetscObjectSetName((PetscObject)X,vecname));
    PetscCall(VecLoad(X,viewer));
    //VecView(X,PETSC_VIEWER_STDOUT_WORLD);

    PetscCall(VecGetArray(X,&_vals));
    for (i=0; i<mlocal; i++) {
      _B[i + j*mlocal] = _vals[i];
    }
    PetscCall(VecRestoreArray(X,&_vals));
  }
  PetscCall(MatDenseRestoreArray(B,&_B));

  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test2(void)
{
  const char attr_dim_name[] = "attr_dim";
  char snapshotprefix[] = "x";
  char fname[] = "tiny.hdf5";
  Mat X;

  PetscFunctionBeginUser;
  PetscCall(MatDenseCreateSnapshotFromH5(PETSC_COMM_WORLD,
    fname,snapshotprefix,attr_dim_name,&X));
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
    PetscCall(test1());
    break;

    case 1:
    PetscCall(test2());
    break;

  }
  PetscCall(PetscFinalize());
  return 0;
}
