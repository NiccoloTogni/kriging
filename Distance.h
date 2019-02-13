
#ifndef DISTANCES_H_
#define DISTANCES_H_

#include "Eigen/Dense" //to define distances matrix
//#include "Eigen/Sparse"                                                       //non serve

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd; //distance matrix

//Euclidean distance
template<typename EuclideanVector>
double distance(const EuclideanVector& p1, const EuclideanVector& p2){
  auto diff = p1-p2;
  double d = diff.norm();
  return d;
}

//Distance matrix
template<typename EuclideanVector>                                              //memory overhead
MatrixXd distance_matrix(const vector<EuclideanVector> &points){                //anche se temporaneo
  if (points.isempty()) return MatrixXd(0,0);
  size_t n = points.size();
  MatrixXd D = MatrixXd::Zero(n,n);
  for (int i=0;i<(n-1);i++)
    for (int j=i+1;j<n;j++)
      D(i,j) = distance(points[i],points[j]);
  return D;
}

#endif
