/* Author: Masaki Murooka */

#pragma once

#include <Dataset.h>
#include <StateEq.h>

namespace DDMPC
{
/*! \brief Class to train a model. */
class Training
{
public:
  /*! \brief Constructor.
      \param device_type type of device (torch::DeviceType::CUDA or torch::DeviceType::CPU)
   */
  Training(torch::DeviceType device_type = torch::DeviceType::CPU);

  /*! \brief Train a model.
      \param train_dataset training dataset
      \param test_dataset test dataset
      \param model_path path to save model parameters
      \param batch_size batch size for train and test
      \param num_epoch nubmer of epoch for learning
   */
  void run(std::shared_ptr<Dataset> & train_dataset,
           std::shared_ptr<Dataset> & test_dataset,
           const std::string & model_path,
           int batch_size = 64,
           int num_epoch = 100);

public:
  //! Device on which to place the model parameters
  std::shared_ptr<torch::Device> device_;

  //! State equation
  std::shared_ptr<StateEq> state_eq_;
};
} // namespace DDMPC
