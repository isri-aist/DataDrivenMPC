/* Author: Masaki Murooka */

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

using namespace DDMPC;

template<int StateDim, int InputDim>
StateEq<StateDim, InputDim>::Model::Model(int middle_dim)
{
  // Instantiate layers
  linear1_ = register_module("linear1", torch::nn::Linear(StateDim + InputDim, middle_dim));
  linear2_ = register_module("linear2", torch::nn::Linear(middle_dim, middle_dim));
  linear3_ = register_module("linear3", torch::nn::Linear(middle_dim, StateDim));

  // Print debug information
  constexpr bool debug = true;
  if(debug)
  {
    std::cout << "Construct NN Module" << std::endl;
    std::cout << "  - state_dim: " << StateDim << std::endl;
    std::cout << "  - input_dim: " << InputDim << std::endl;
    std::cout << "  - layer dims: " << StateDim + InputDim << " -> " << middle_dim << " -> " << StateDim << std::endl;
  }
}

template<int StateDim, int InputDim>
torch::Tensor StateEq<StateDim, InputDim>::Model::forward(torch::Tensor & x, torch::Tensor & u)
{
  // Check dimensions
  assert(x.size(1) == StateDim);
  assert(u.size(1) == InputDim);

  // Calculate network output
  torch::Tensor xu = torch::cat({x, u}, 1);
  xu = torch::relu(linear1_(xu));
  xu = torch::relu(linear2_(xu));
  return linear3_(xu);
}

template<int StateDim, int InputDim>
typename StateEq<StateDim, InputDim>::StateDimVector StateEq<StateDim, InputDim>::calc(const StateDimVector & x,
                                                                                       const InputDimVector & u)
{
  torch::Tensor x_tensor = toTorchTensor(x.transpose().template cast<float>());
  torch::Tensor u_tensor = toTorchTensor(u.transpose().template cast<float>());
  torch::Tensor next_x_tensor = model_ptr_->forward(x_tensor, u_tensor);
  return toEigenMatrix(next_x_tensor).template cast<double>();
}
