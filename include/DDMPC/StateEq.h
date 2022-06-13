/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>
#include <torch/torch.h>

namespace DDMPC
{
/*! \brief Class of state equation based on neural network model.
    \tparam StateDim state dimension
    \tparam InputDim input dimension
*/
template<int StateDim, int InputDim>
class StateEq
{
public:
  /** \brief Type of vector of state dimension. */
  using StateDimVector = Eigen::Matrix<double, StateDim, 1>;

  /** \brief Type of vector of input dimension. */
  using InputDimVector = Eigen::Matrix<double, InputDim, 1>;

  /*! \brief Class of neural network model for state equation. */
  class Model : public torch::nn::Module
  {
  public:
    /*! \brief Constructor.
        \param middle_dim dimension of middle layer
     */
    Model(int middle_dim = 100);

    /*! \brief Forward model.
        \param x current state
        \param u current input
        \returns next state
     */
    torch::Tensor forward(torch::Tensor & x, torch::Tensor & u);

  public:
    //! Linear layers
    torch::nn::Linear linear1_ = nullptr;
    torch::nn::Linear linear2_ = nullptr;
    torch::nn::Linear linear3_ = nullptr;
  };

  /*! \brief Model pointer.

      See "Module Ownership" section of https://pytorch.org/tutorials/advanced/cpp_frontend.html for details
   */
  TORCH_MODULE_IMPL(ModelPtr, Model);

public:
  /*! \brief Constructor. */
  StateEq() : model_ptr_(ModelPtr()) {}

  /*! \brief Calculate next state.
      \param x current state
      \param u current input
      \returns next state
   */
  StateDimVector calc(const StateDimVector & x, const InputDimVector & u);

public:
  //! Model pointer
  ModelPtr model_ptr_ = nullptr;
};
} // namespace DDMPC
