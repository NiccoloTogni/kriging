
#ifndef ORDINARYKRIGING_H_
#define ORDINARYKRIGING_H_

#include<iostream> //error messages
#include<vector>
#include<limits> //for quiet_NaN
//#include<memory> //we use shared_ptrs to avoid doubling the data
#include "Eigen/Dense" //Linear algebra
#include "Variogram.h"
#include "Distance.h"

using namespace std;
using namespace Eigen;

//Isotropic Ordinary kriging with scalar output

template<typename position_type>
class OrdinaryKriging {                                                         //valutare se estendere
public:
  typedef Matrix<double, Dynamic, Dynamic> MatrixXd;  //for the covariance matrix
  typedef Matrix<double, Dynamic, 1> VectorXd;       //zs, prediction coefficients
  typedef position_type PT;  //to simplify the notation
  typedef shared_ptr<const vector<PT>> locs_ptr;
  typedef shared_ptr<const VectorXd> vals_ptr;

  friend class Variogram;
private:
  //data
  locs_ptr locations;
  vals_ptr zs;
  size_t nobs=0; //dimension of the sample

  bool fitted = 0;   //can't use .predict() unless we fitted to some data
  size_t nfit=0; //nr of data used to fit the model
  vector<unsigned> fit_indices = {}; //useful for bootstrap/CrossValidation/jackknifing

  double mu = 0;  //empirical mean of the process
  VectorXd prediction_coeffs; //coefficients for linear prediction
  Variogram vg; //variogram that we'll have to fit to the data

  void fit__();  //private function used to fit
  bool has_data() const {return nobs > 0;}

public:
  OrdinaryKriging() = default;
  OrdinaryKriging(const vector<PT>& locs, const VectorXd& values);
  //When we initialize the kriging we don't fit it immediately
  void set_data(const vector<PT>& locs, const VectorXd& values);

  void fit();
  void fit(const vector<unsigned>& indices);  //specialized function that fits using only a subset of the sample
  void loo_fit(const unsigned& leave_out); //specialized function for loo-cv/jackknife
  //when whe fit we fit the covariance model to the data, then we
  //compute Sigma, mu and the coefficients of prediction.

  double predict(const PT& new_loc);
  VectorXd predict(const vector<PT>& new_locs);
  // We don't compute the variance of the output in the classical way because we will
  // estimate it using cross validation or bootstrap

  //Getters
  bool is_fitted() const {return fitted;}
  size_t size() const {return nobs;}
  double mean() const {return mu;}
  Variogram get_variogram() const {return vg;}
  vector<PT>& get_locations() {return *locations;} //getters for the data
  VectorXd& get_values() {return *zs;}

};

/*----------------------------------------------------------------------------*/
/*----------------------------  Definitions  ---------------------------------*/
/*----------------------------------------------------------------------------*/

//Constructor
template<typename PT>
OrdinaryKriging<PT>::OrdinaryKriging(const vector<PT>& locs, const VectorXd& values):
  locations(&locs),zs(&values){                                                 // MODIFICA
  if (locs.size() != values.size()){
    cerr<<"The number of locations doesn't correspond to the number of observed values\n";
    return;
  }
  else {
    nobs = values.size();
    return;
  }
}

//Set data
template<typename PT>
void OrdinaryKriging<PT>::set_data(const vector<PT>& locs, const VectorXd& values){
  if (has_data()){
    if((locations.get() == &locs) && (zs.get() == &values)) return;
  }
  size_t temp_n = values.size();
  if (locs.size() != temp_n){
    cerr<<"The number of locations doesn't correspond to the number of observed values\n";
    return;
  }
  locations = make_shared<const vector<PT>>(locs);                              // MODIFICA
  zs = make_shared<const VectorXd>(values);
  nobs = locations->size();
  return;
}

//fit functions
template<typename PT>
void OrdinaryKriging<PT>::fit__(){
  VectorXd fit_values(nfit);
  vector<PT> fit_locs(nfit);
  for(int i=0;i<nfit;i++) {
    fit_values(i) = (*zs)(fit_indices[i]);
    fit_locs[i] = (*locations)[fit_indices[i]];
  }
  //vg.fit(fit_locs,fit_values); //fit covariance function to the data
  //Compute Sigma
  MatrixXd Sigma(nfit,nfit);                                                    //unnecessary to store it
  for (int i=0;i<nfit;i++){
    for (int j=i;j<nfit;j++){
      Sigma(i,j) = vg.get_sill()-vg.compute(distance(fit_locs[i],fit_locs[j]));
    }
  }
  for(int i=1;i<nfit;i++){                                                      //possibile usare matrice sparsa
    for(int j=0;j<i;j++){
    Sigma(i,j) = Sigma(j,i);
    }
  }
  //cout<<"Sigma is\n"<<Sigma<<endl;                                            //controllo codice
  VectorXd ones(nfit);
  for (int i=0;i<nfit;i++)
    ones(i) = 1;
  mu = ones.dot(Sigma.ldlt().solve(fit_values))/(ones.dot(Sigma.ldlt().solve(ones)));  //valutare parallelizzazione
  prediction_coeffs = Sigma.ldlt().solve(fit_values-mu*ones);                          //o algoritmi più efficienti
  fitted = 1;
  return;
}

template<typename PT>
void OrdinaryKriging<PT>::fit(){
  if(! has_data()){
    cerr<<"Need valid data to fit\n";
    return;
  }
  if(fitted){
    return;
  }
  nfit = nobs; //we fit using all available data with no resampling
  fit_indices = vector<unsigned>(nfit,0);                                       //spreco di memoria ma difficile fare meglio
  for(int i=0;i<nfit;i++) fit_indices[i] = i;
  fit__();
  return;
}

template<typename PT>
void OrdinaryKriging<PT>::fit(const vector<unsigned>& indices){
  if(! has_data()){
    cerr<<"Need valid data to fit\n";
    return;
  }
  if(fitted){
    if(fit_indices==indices) return;
  }
  nfit = indices.size();
  if (nfit > nobs){
    cerr<<"Numer of indices exceeds the dimension of the sample"<<endl;
    nfit = 0;
    return;
  }
  fit_indices = indices;
  fit__();
  return;
}

template<typename PT>
void OrdinaryKriging<PT>::loo_fit(const unsigned& leave_out){
  if(! has_data()){
    cerr<<"Need valid data to fit\n";
    return;
  }
  if(fitted){
    if(fit_indices.size() == (nobs-1)){                                         //migliorabile
      bool flag=0;
      for(auto ix: fit_indices)
        if(ix == leave_out)
          flag = 1;
      if(flag) return;
    }
  }
  nfit = nobs-1;
  fit_indices = {};
  for(int i=0;i<nobs;i++){
    if (i != leave_out) fit_indices.push_back(i);
  }
  fit__();
  return;
}

//predict functions
template<typename PT>
double OrdinaryKriging<PT>::predict(const PT& new_loc){
  if (!fitted){
    cerr<<"Model has not been fitted yet\n";
    return numeric_limits<double>::quiet_NaN();
  }
  VectorXd cov_vect(nfit);
  for(int i=0;i<nfit;i++){
    cov_vect(i) = vg.get_sill() - vg.compute( distance( (*locations)[fit_indices[i]],new_loc ) );
  }
  return mu + cov_vect.dot(prediction_coeffs);
  // mu + gamma'* Sigma^-1(w-mu*1)
}

//predict vector of new data
template<typename PT>
VectorXd OrdinaryKriging<PT>::predict(const vector<PT>& new_locs){
  if (!fitted){
    cerr<<"Model has not been fitted yet\n";
    double nan = numeric_limits<double>::quiet_NaN();
    VectorXd v = VectorXd::Zero(new_locs.size());
    return v;
  }
  size_t n_new = new_locs.size();
  VectorXd predicted_values(n_new);
  int i = 0;
  for(auto new_loc: new_locs){
    predicted_values[i] = predict(new_loc);
    i++;
  }
  return predicted_values;
}

#endif
