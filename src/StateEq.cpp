/* Author: Masaki Murooka */

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

using namespace DDMPC;

StateEq::Model::Model(int state_dim, int input_dim, int middle_dim) : state_dim_(state_dim), input_dim_(input_dim)
{
  // Instantiate layers
  linear1_ = register_module("linear1", torch::nn::Linear(state_dim_ + input_dim_, middle_dim));
  linear2_ = register_module("linear2", torch::nn::Linear(middle_dim, middle_dim));
  linear3_ = register_module("linear3", torch::nn::Linear(middle_dim, state_dim_));

  // Print debug information
  if(debug_)
  {
    std::cout << "Construct NN Module" << std::endl;
    std::cout << "  - state_dim: " << state_dim_ << std::endl;
    std::cout << "  - input_dim: " << input_dim_ << std::endl;
    std::cout << "  - layer dims: " << state_dim_ + input_dim_ << " -> " << middle_dim << " -> " << state_dim_
              << std::endl;
  }
}

torch::Tensor StateEq::Model::forward(torch::Tensor & x, torch::Tensor & u)
{
  // Check dimensions
  assert(x.size(1) == state_dim_);
  assert(u.size(1) == input_dim_);

  // Calculate network output
  torch::Tensor xu = torch::cat({x, u}, 1);
  xu = torch::relu(linear1_(xu));
  xu = torch::relu(linear2_(xu));
  return linear3_(xu);
}

typename Eigen::VectorXd StateEq::calc(const Eigen::VectorXd & x, const Eigen::VectorXd & u)
{
  // Check dimensions
  assert(x.size() == stateDim());
  assert(u.size() == inputDim());

  // Calculate network output
  torch::Tensor x_tensor = toTorchTensor(x.transpose().template cast<float>());
  torch::Tensor u_tensor = toTorchTensor(u.transpose().template cast<float>());
  torch::Tensor next_x_tensor = model_ptr_->forward(x_tensor, u_tensor);
  return toEigenMatrix(next_x_tensor).template cast<double>();
}
