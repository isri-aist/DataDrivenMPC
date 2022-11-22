/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>
#include <torch/torch.h>

namespace DDMPC
{
/** \brief Class of state equation based on neural network model. */
class StateEq
{
public:
  /** \brief Class of neural network model for state equation. */
  class Model : public torch::nn::Module
  {
  public:
    /** \brief Constructor.
        \param state_dim state dimension
        \param input_dim input dimension
        \param middle_layer_dim middle layer dimension
     */
    Model(int state_dim, int input_dim, int middle_layer_dim);

    /** \brief Forward model.
        \param x current state
        \param u current input
        \param enable_auto_grad whether to enable automatic gradient (default true)
        \param requires_grad_x whether to require gradient w.r.t. current state (default false)
        \param requires_grad_u whether to require gradient w.r.t. current input (default false)
        \returns next state

        The required gradients are stored in the member variables grad_x_ and grad_u_.
     */
    torch::Tensor forward(torch::Tensor & x,
                          torch::Tensor & u,
                          bool enable_auto_grad = true,
                          bool requires_grad_x = false,
                          bool requires_grad_u = false);

  public:
    //! Whether to enable debug print
    bool debug_ = true;

    //! State dimension
    const int state_dim_;

    //! Input dimension
    const int input_dim_;

    //! Linear layers
    torch::nn::Linear linear1_ = nullptr;
    torch::nn::Linear linear2_ = nullptr;
    torch::nn::Linear linear3_ = nullptr;

    //! Gradient w.r.t. current state
    torch::Tensor grad_x_;

    //! Gradient w.r.t. current input
    torch::Tensor grad_u_;
  };

  /** \brief Model pointer.

      See "Module Ownership" section of https://pytorch.org/tutorials/advanced/cpp_frontend.html for details
   */
  TORCH_MODULE_IMPL(ModelPtr, Model);

public:
  /** \brief Constructor.
      \param state_dim state dimension
      \param input_dim input dimension
      \param middle_layer_dim middle layer dimension
   */
  StateEq(int state_dim, int input_dim, int middle_layer_dim = 32)
  : model_ptr_(ModelPtr(state_dim, input_dim, middle_layer_dim))
  {
  }

  /** \brief Calculate next state.
      \param x current state
      \param u current input
      \returns next state
   */
  Eigen::VectorXd eval(const Eigen::VectorXd & x, const Eigen::VectorXd & u);

  /** \brief Calculate next state.
      \param x current state
      \param u current input
      \param[out] grad_x gradient w.r.t. x (not calculated when the matrix size is zero)
      \param[out] grad_u gradient w.r.t. u (not calculated when the matrix size is zero)
      \returns next state
   */
  Eigen::VectorXd eval(const Eigen::VectorXd & x,
                       const Eigen::VectorXd & u,
                       Eigen::Ref<Eigen::MatrixXd> grad_x,
                       Eigen::Ref<Eigen::MatrixXd> grad_u);

  /** \brief Get state dimension. */
  inline int stateDim() const
  {
    return model_ptr_->state_dim_;
  }

  /** \brief Get input dimension. */
  inline int inputDim() const
  {
    return model_ptr_->input_dim_;
  }

public:
  //! Model pointer
  ModelPtr model_ptr_ = nullptr;
};
} // namespace DDMPC
