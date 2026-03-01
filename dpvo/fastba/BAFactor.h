#include <torch/extension.h>
#include <vector>
#include "block_e.cuh"

class BAFactor
{
public:
  BAFactor(){}
  ~BAFactor(){}
public:
  void init(torch::Tensor poses,
            torch::Tensor patches,
            torch::Tensor intrinsics,
            torch::Tensor target,
            torch::Tensor weight,
            torch::Tensor lmbda,
            torch::Tensor ii,
            torch::Tensor jj, 
            torch::Tensor kk,
            int PPF,
            int t0, int t1, int iterations, bool eff_impl);
  void hessian(torch::Tensor Hgg, torch::Tensor vgg);
  std::vector<torch::Tensor> retract(torch::Tensor _dx);
  
public:
  torch::Tensor poses;
  torch::Tensor patches;
  torch::Tensor intrinsics;
  torch::Tensor target;
  torch::Tensor weight;
  torch::Tensor lmbda;
  torch::Tensor ii;
  torch::Tensor jj;
  torch::Tensor kk;
  int PPF;//PATCHES_PER_FRAME
  int t0,t1;

  bool eff_impl;//=false;

  torch::Tensor kx;
  torch::Tensor ku;

  // initialize buffers
  std::unique_ptr<EfficentE> blockE;
  torch::Tensor B;
  torch::Tensor E;
  torch::Tensor C;//一个对角矩阵

  torch::Tensor v;
  torch::Tensor u;


};