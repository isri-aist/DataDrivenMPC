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
    std::cout << "  - layer dims: " << state_dim_ + input_dim_ << " -> " << middle_layer_dim << " -> " << state_dim_
              << std::endl;
  }

  // Workaround to avoid torch error
  // See https://github.com/pytorch/pytorch/issues/35736#issuecomment-688078143
  torch::cuda::is_available();
}

torch::Tensor StateEq::Model::forward(torch::Tensor & x, torch::Tensor & u)
{
  torch::Tensor grad_x = torch::empty({});
  torch::Tensor grad_u = torch::empty({});
  return forward(x, u, grad_x, grad_u);
}

torch::Tensor StateEq::Model::forward(torch::Tensor & x,
                                      torch::Tensor & u,
                                      torch::Tensor & grad_x,
                                      torch::Tensor & grad_u)
{
  // Check dimensions
  assert(x.size(1) == state_dim_);
  assert(u.size(1) == input_dim_);
  assert(x.size(0) == u.size(0));
  assert(grad_x.dim() == 0 || (grad_x.size(0) == state_dim_ && grad_x.size(1) == state_dim_));
  assert(grad_u.dim() == 0 || (grad_u.size(0) == state_dim_ && grad_u.size(1) == input_dim_));

  // Setup gradient
  bool requires_grad_x = (grad_x.dim() > 0);
  x.set_requires_grad(requires_grad_x);
  if(requires_grad_x)
  {
    auto & x_grad = x.mutable_grad();
    if(x_grad.defined())
    {
      x_grad = x_grad.detach();
      x_grad.zero_();
    }
  }
  bool requires_grad_u = (grad_u.dim() > 0);
  u.set_requires_grad(requires_grad_u);
  if(requires_grad_u)
  {
    auto & u_grad = u.mutable_grad();
    if(u_grad.defined())
    {
      u_grad = u_grad.detach();
      u_grad.zero_();
    }
  }
  if((requires_grad_x || requires_grad_u) && x.size(0) > 1)
  {
    throw std::runtime_error("batch size should be 1 when requiring gradient. batch size: "
                             + std::to_string(x.size(0)));
  }

  // Calculate network output
  torch::Tensor xu = torch::cat({x, u}, 1);
  xu = torch::relu(linear1_(xu));
  xu = torch::relu(linear2_(xu));
  torch::Tensor next_x = linear3_(xu);

  // Calculate gradient
  if(requires_grad_x || requires_grad_u)
  {
    for(int i = 0; i < state_dim_; i++)
    {
      // Calculate backward of each element
      torch::Tensor select = torch::zeros({1, state_dim_});
      select.index({0, i}) = 1;
      next_x.backward(select, true);

      // Set gradient
      if(requires_grad_x)
      {
        grad_x.index({i}) = x.grad().index({0});
        x.mutable_grad().zero_();
      }
      if(requires_grad_u)
      {
        grad_u.index({i}) = u.grad().index({0});
        u.mutable_grad().zero_();
      }
    }
  }

  return next_x;
}

Eigen::VectorXd StateEq::eval(const Eigen::VectorXd & x, const Eigen::VectorXd & u)
{
  Eigen::MatrixXd grad_x = Eigen::MatrixXd::Zero(0, 0);
  Eigen::MatrixXd grad_u = Eigen::MatrixXd::Zero(0, 0);
  return eval(x, u, grad_x, grad_u);
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
  torch::Tensor grad_x_tensor = grad_x.size() > 0 ? torch::empty({grad_x.rows(), grad_x.cols()}) : torch::empty({});
  torch::Tensor grad_u_tensor = grad_u.size() > 0 ? torch::empty({grad_u.rows(), grad_u.cols()}) : torch::empty({});

  // Forward network
  torch::Tensor next_x_tensor = model_ptr_->forward(x_tensor, u_tensor, grad_x_tensor, grad_u_tensor);

  // Set output variables
  if(grad_x.size() > 0)
  {
    grad_x = toEigenMatrix(grad_x_tensor).cast<double>();
  }
  if(grad_u.size() > 0)
  {
    grad_u = toEigenMatrix(grad_u_tensor).cast<double>();
  }
  return toEigenMatrix(next_x_tensor).transpose().cast<double>();
}
