
#ifndef JACK_KNIFE_H_
#define JACK_KNIFE_H_

#include<iostream> //error messages
#include<vector>
#include<mpi.h>
#include "Eigen/Dense"

typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<double, 1, 2> mean_and_variance;                                 //non trova "pair"

template<typename predictor, typename input_data>
vector<mean_and_variance> ParallelJackKnifeEstimator (predictor kr, vector<input_data> new_data){
  unsigned N = kr.size();
  if( N == 0 ){
    cerr<<"Invalid input parameters: model must be fitted\n";
    return vector<mean_and_variance>();
  }
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  unsigned local_share = N / size;
  if (rank < N % size) ++local_share;

  vector<predictor> local_models(local_share);
  //train models
  int aux = 0;
  for(int i=rank*local_share;i<(rank+1)*local_share;i++){                       //valutare se cambiare
    local_models[aux] = kr; //copies data
    //cout<<"process "<<rank<<" fit -"<<i<<"\n";
    local_models[aux].loo_fit(i);                                               //valutare se rimuovere loo_fit
    aux++;
  }

  //Each process fits only a fraction of jackknifed models
  vector<mean_and_variance> result;

  kr.fit(); //fit model using all data                                          //preferibile chiamare fit() prima
  for(auto new_loc: new_data){                                                  //di passare il predittore a jackknife
                                                                                //altrimenti ogni processo fa calcoli superflui
    vector<double> y_hat(local_share);
    double y_mean=0.;

    for(int i=0;i<local_share;i++){
      y_hat[i] = local_models[i].predict(new_loc);
      y_mean += y_hat[i];
    } //vector of predictions

    MPI_Allreduce (MPI_IN_PLACE, &y_mean, 1, MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD);
    y_mean = y_mean/N;

    //Finally we compute the variance
    double variance = 0.;
    for (double y: y_hat) variance += (y - y_mean) * (y - y_mean);

    MPI_Allreduce (MPI_IN_PLACE, &variance, 1, MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD);
    variance  = variance*(N - 1)/N;  //jacknifed variance

    mean_and_variance mv;
    //jackknife bias correction
    double full_pred = kr.predict(new_loc);
    if (rank == 0) {
      //cout<<"prediction coeffs = \n"<<kr.get_prediction_coeff()<<endl;
      cout<<"prediction full model = "<<full_pred<<"\n"
          <<"jackknife mean = "<<y_mean<<"\n";
    }
    mv[0] = N*full_pred - (N-1)*y_mean; //unbiased jackknife estimate
    mv[1] = variance;
    result.push_back(mv);
  }
  return result;
}

//If we pass the data as a parameter
template<typename predictor, typename input_data>
vector<mean_and_variance> ParallelJackKnifeEstimator (predictor kr,
  const vector<input_data>& X_train, const VectorXd& y_train,
  vector<input_data> new_data) {

  if(X_train.size() != y_train.size() ){
    cerr<<"Invalid data\n";
    return vector<mean_and_variance>();
  }
  else {
    kr.set_data(X_train,y_train);
    return ParallelJackKnifeEstimator(kr,new_data);
  } //else
}


#endif
