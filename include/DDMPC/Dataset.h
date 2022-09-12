/* Author: Masaki Murooka */

#pragma once

#include <memory>

#include <torch/torch.h>

namespace DDMPC
{
/** \brief Class of single data for state equation. */
class Data
{
public:
  /** \brief Constructor.
      \param state single data tensor of current state
      \param input single data tensor of current input
      \param next_state single data tensor of next state
   */
  Data(const torch::Tensor & state, const torch::Tensor & input, const torch::Tensor & next_state)
  : state_(state), input_(input), next_state_(next_state)
  {
  }

public:
  //! Single data tensor of current state
  torch::Tensor state_;

  //! Single data tensor of current input
  torch::Tensor input_;

  //! Single data tensor of next state
  torch::Tensor next_state_;
};

/** \brief Single data example of state equation. */
using Example = torch::data::Example<Data, torch::data::example::NoTarget>;

/** \brief Class of dataset for state equation. */
class Dataset : public torch::data::datasets::Dataset<Dataset, Example>
{
public:
  /** \brief Constructor.
      \param state all data tensor of current state
      \param input all data tensor of current input
      \param next_state all data tensor of next state
   */
  explicit Dataset(const torch::Tensor & state, const torch::Tensor & input, const torch::Tensor & next_state)
  : state_(std::move(state)), input_(std::move(input)), next_state_(std::move(next_state))
  {
    // Store size because tensors are passed to data loader with std::move
    size_ = state_.size(0);
  }

  /** \brief Returns a single data example. */
  inline Example get(size_t index) override
  {
    return Data(state_[index], input_[index], next_state_[index]);
  }

  /** \brief Returns dataset size. */
  inline torch::optional<size_t> size() const override
  {
    return size_;
  }

protected:
  //! All data tensor of current state
  torch::Tensor state_;

  //! All data tensor of current input
  torch::Tensor input_;

  //! All data tensor of next state
  torch::Tensor next_state_;

  //! Dataset size
  size_t size_;
};

/** \brief Make dataset.
    \param state all data tensor of current state
    \param input all data tensor of current input
    \param next_state all data tensor of next state
    \param[out] train_dataset training dataset
    \param[out] test_dataset test dataset
 */
void makeDataset(const torch::Tensor & state,
                 const torch::Tensor & input,
                 const torch::Tensor & next_state,
                 std::shared_ptr<Dataset> & train_dataset,
                 std::shared_ptr<Dataset> & test_dataset);

/** \brief Make batch tensors.
    \param batch batch from data loader
    \param device device to which tensors belong
    \param[out] b_state batch tensor of current state
    \param[out] b_next_state batch tensor of next state
    \param[out] b_input batch tensor of input
 */
void makeBatchTensor(const std::vector<Example> & batch,
                     const torch::Device & device,
                     torch::Tensor & b_state,
                     torch::Tensor & b_input,
                     torch::Tensor & b_next_state);
} // namespace DDMPC
