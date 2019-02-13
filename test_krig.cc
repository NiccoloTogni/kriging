
#include "Eigen/Dense"
#include "Variogram.h"
#include "Kriging.h"
#include "ReadData.h"
#include<iostream>
#include<vector>

using namespace std;
using namespace Eigen;

int main(){

  typedef Matrix<double, Dynamic, 1> VectorXd;
  //Kriging 2d
  typedef Matrix<double, 1, 2> position;  //locations are 2D vectors in this example
  //typedef ExponentialCovariance<position> covariance; //(...)

  const string Xfile = "data/locs.txt";
  const string yfile = "data/vals.txt";

  vector<position> locs;
  readX(locs,Xfile);

  /*cout<<"X=\n"<<locs[0]<<"\n\n"<<locs[1]<<"\n\n"
                               <<locs[2]<<"\n\n"
                               <<locs[3]<<"\n\n";*/
  VectorXd ys;
  ready(ys,yfile);

  //cout<<"y\n"<<ys<<endl;

  OrdinaryKriging<position> kr;
  //cout<<"Kriging initialization OK"<<endl;

  kr.set_data(locs,ys);
  //std::cout << "set_data OK" << '\n';

  kr.fit();
  //std::cout << "fit OK" << '\n';

  auto m = kr.mean();
  //cout<<"estimated mu = "<<m<<endl;

  //auto coeff = kr.get_prediction_coeff();
  //cout<<"prediction coefficients = \n"<<coeff<<endl;

  vector<position> new_locs(6);
  new_locs[0] << 0.5, 0;
  new_locs[1] << 0, 0.5;
  new_locs[2] << 0.5, 0.5;
  new_locs[3] << 0.2, 0.3;
  new_locs[4] << 0.7, 0.4;
  new_locs[5] << 0.1, 0.9;

  auto preds = kr.predict(new_locs);

  cout<<"predicted values\n"
      <<preds[0]<<"\n"
      <<preds[1]<<"\n"
      <<preds[2]<<"\n"
      <<preds[3]<<"\n"
      <<preds[4]<<"\n"
      <<preds[5]<<"\n";

  return 0;
}
