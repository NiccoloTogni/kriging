
#ifndef VARIOGRAM_H_
#define VARIOGRAM_H_

#include<math.h>
#include "Distance.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

//Exponential covariance  without nugget(useful when we have a great amount of data)
//cov(d) = sill*exp{-d/theta}

class Variogram {
public:
  typedef Matrix<double, Dynamic, Dynamic> MatrixXd; //distance matrix
  typedef Matrix<double, Dynamic, 1> VectorXd;
private:
  double sill = 2.;  //sigma^2
  double theta = 1.; //slope
  double range = 10.; //range
  double nugget = 0.; //jump in 0

public:
  //void fit(const vector<position>& locs, const VectorXd& ys){return;}
  double compute(const double& d){return sill*(1-exp(-d/theta));}

  double get_sill() const {return sill;}
  double get_slope() const {return theta;}
  double get_range() const {return range;}
  double get_nugget() const {return nugget;}
};

#endif
