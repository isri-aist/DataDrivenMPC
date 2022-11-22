/* Author: Masaki Murooka */

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

using namespace DDMPC;

StateEq::Model::Model(int state_dim, int input_dim, int middle_layer_dim) : state_dim_(state_dim), input_dim_(input_dim)
{
  // Instantiate layers
  linear1_ = register_module("linear1", torch::nn::Linear(state_dim_ + input_dim_, middle_layer_dim));
  linear2_ = register_module("linear2", torch::nn::Linear(middle_layer_dim, middle_layer_dim));
  linear3_ = register_module("linear3", torch::nn::Linear(middle_layer_dim, state_dim_));

  // Print debug information
  if(debug_)
  {
    std::cout << "Construct NN Module" << std::endl;
    std::cout << "  - state_dim: " << state_dim_ << std::endl;
    std::cout << "  - input_dim: " << input_dim_ << std::endl;
    std::cout << "  - layer dims: " << state_dim_ + input_dim_ << " -> " << middle_layer_dim << " -> "
              << middle_layer_dim << " -> " << state_dim_ << std::endl;
  }

  // Workaround to avoid torch error
  // See https://github.com/pytorch/pytorch/issues/35736#issuecomment-688078143
  torch::cuda::is_available();
}

torch::Tensor StateEq::Model::forward(torch::Tensor & x,
                                      torch::Tensor & u,
                                      bool enable_auto_grad,
                                      bool requires_grad_x,
                                      bool requires_grad_u)
{
  // Check dimensions
  assert(x.size(1) == state_dim_);
  assert(u.size(1) == input_dim_);
  assert(x.size(0) == u.size(0));

  // Setup gradient
  bool requires_grad = (requires_grad_x || requires_grad_u);
  if(requires_grad)
  {
    enable_auto_grad = true;
  }
  torch::Tensor x_repeated;
  torch::Tensor u_repeated;
  if(requires_grad)
  {
    x_repeated = x.repeat({state_dim_, 1});
    u_repeated = u.repeat({state_dim_, 1});
  }
  if(requires_grad_x)
  {
    x_repeated.set_requires_grad(true);
    assert(!x_repeated.mutable_grad().defined());
  }
  if(requires_grad_u)
  {
    u_repeated.set_requires_grad(true);
    assert(!u_repeated.mutable_grad().defined());
  }
  if(requires_grad && x.size(0) > 1)
  {
    throw std::runtime_error("batch size should be 1 when requiring gradient. batch size: "
                             + std::to_string(x.size(0)));
  }

  std::unique_ptr<torch::NoGradGuard> no_grad;
  if(!enable_auto_grad)
  {
    // Not calculate gradient
    no_grad = std::make_unique<torch::NoGradGuard>();
  }

  // Calculate network output
  torch::Tensor xu = requires_grad ? torch::cat({x_repeated, u_repeated}, 1) : torch::cat({x, u}, 1);
  xu = torch::relu(linear1_(xu));
  xu = torch::relu(linear2_(xu));
  torch::Tensor next_x = linear3_(xu);

  // Calculate gradient
  if(requires_grad)
  {
    // See https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa for Jacobian calculation
    next_x.backward(torch::eye(state_dim_));
    if(requires_grad_x)
    {
      grad_x_ = x_repeated.grad();
    }
    if(requires_grad_u)
    {
      grad_u_ = u_repeated.grad();
    }
  }

  return requires_grad ? next_x.index({0}).view({1, -1}) : next_x;
}

Eigen::VectorXd StateEq::eval(const Eigen::VectorXd & x, const Eigen::VectorXd & u)
{
  // Check dimensions
  assert(x.size() == stateDim());
  assert(u.size() == inputDim());

  // Set tensor
  torch::Tensor x_tensor = toTorchTensor(x.transpose().cast<float>());
  torch::Tensor u_tensor = toTorchTensor(u.transpose().cast<float>());

  // Forward network
  torch::Tensor next_x_tensor = model_ptr_->forward(x_tensor, u_tensor, false);

  // Set output variables
  return toEigenMatrix(next_x_tensor).transpose().cast<double>();
}

Eigen::VectorXd StateEq::eval(const Eigen::VectorXd & x,
                              const Eigen::VectorXd & u,
                              Eigen::Ref<Eigen::MatrixXd> grad_x,
                              Eigen::Ref<Eigen::MatrixXd> grad_u)
{
  // Check dimensions
  assert(x.size() == stateDim());
  assert(u.size() == inputDim());
  assert(grad_x.size() == 0 || (grad_x.rows() == stateDim() && grad_x.cols() == stateDim()));
  assert(grad_u.size() == 0 || (grad_u.rows() == stateDim() && grad_u.cols() == inputDim()));

  // Set tensor
  torch::Tensor x_tensor = toTorchTensor(x.transpose().cast<float>());
  torch::Tensor u_tensor = toTorchTensor(u.transpose().cast<float>());
  bool requires_grad_x = grad_x.size() > 0;
  bool requires_grad_u = grad_u.size() > 0;

  // Forward network
  torch::Tensor next_x_tensor = model_ptr_->forward(x_tensor, u_tensor, false, requires_grad_x, requires_grad_u);

  // Set output variables
  if(requires_grad_x)
  {
    grad_x = toEigenMatrix(model_ptr_->grad_x_).cast<double>();
  }
  if(requires_grad_u)
  {
    grad_u = toEigenMatrix(model_ptr_->grad_u_).cast<double>();
  }
  return toEigenMatrix(next_x_tensor).transpose().cast<double>();
}
