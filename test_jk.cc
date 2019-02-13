
#include<iostream>
#include<vector>
#include<mpi.h>
#include "Eigen/Dense"
#include "Kriging.h"
#include "ReadData.h"
#include "JackKnife.h"

using namespace std;
using namespace Eigen;


/*---------------Test JackKnife in parallel---------------*/

int main(int argc, char *argv[]) {


  MPI_Init (&argc, &argv);

  typedef Matrix<double, Dynamic, 1> VectorXd;
  //Kriging 2d
  typedef Matrix<double, 1, 2> position;  //locations are 2D vectors in this example
  //typedef ExponentialCovariance<position> covariance; //(...)

  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  //We read the input data from process 0 and broadcast them to every process
  //Then we just call ParallelJackKnifeEstimator

  unsigned N (0);
  unsigned x_dim (0);   //dimension of input data;

  vector<position> locs;
  VectorXd ys;

  if (rank == 0) {
    const string Xfile = "data/locs.txt";
    const string yfile = "data/vals.txt";
    readX(locs,Xfile);
    ready(ys,yfile);
    N = ys.size();
    x_dim = locs[0].size();
  }

  MPI_Bcast (&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast (&x_dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (rank != 0){
    locs = vector<position>(N);
    ys = VectorXd(N);
  }

  for (int i=0;i<N;i++){
    //Broadcast ys is straightforward
    MPI_Bcast (&ys[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  for (int i=0;i<N;i++){
    //Broadcast X
    for(int j=0;j<x_dim;j++){
        MPI_Bcast (&(locs[i](j)), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  //test
  VectorXd ref_y(4);
  ref_y << 0, 2, 1.5, 0.5;

  vector<position> ref_locs(4);
  ref_locs[0] << 0, 0;
  ref_locs[1] << 0, 1;
  ref_locs[2] << 1, 0;
  ref_locs[3] << 1, 1;

  int gud = 1;
  if ((locs != ref_locs) || (ys != ref_y)) gud = 0;

  MPI_Allreduce (MPI_IN_PLACE, &gud, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank==0) if (gud) cout<<"data broadcasted succesfully\n\n";

  //test jackknife function
  vector<position> new_locs(6);
  new_locs[0] << 0.5, 0;
  new_locs[1] << 0, 0.5;
  new_locs[2] << 0.5, 0.5;
  new_locs[3] << 0.2, 0.3;
  new_locs[4] << 0.7, 0.4;
  new_locs[5] << 0.1, 0.9;

  OrdinaryKriging<position> krige(locs,ys);
  auto result = ParallelJackKnifeEstimator(krige,new_locs);

  if (rank == 0 ) cout<<"media - varianza\n"
                      <<result[0][0]<<" - "<<result[0][1]<<"\n"
                      <<result[1][0]<<" - "<<result[1][1]<<"\n"
                      <<result[2][0]<<" - "<<result[2][1]<<"\n"
                      <<result[3][0]<<" - "<<result[3][1]<<"\n"
                      <<result[4][0]<<" - "<<result[4][1]<<"\n"
                      <<result[5][0]<<" - "<<result[5][1]<<"\n";

  MPI_Finalize();
  return 0;
}
