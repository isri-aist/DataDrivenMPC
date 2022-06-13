/* Author: Masaki Murooka */

#include <DDMPC/TorchUtils.h>
#include <DDMPC/Training.h>

using namespace DDMPC;

Training::Training(torch::DeviceType device_type)
{
  // Set device
  if(!torch::cuda::is_available() && device_type == torch::DeviceType::CUDA)
  {
    std::cout << "CUDA is unavailable! Overwrite with CPU" << std::endl;
    device_type = torch::DeviceType::CPU;
  }
  device_ = std::make_shared<torch::Device>(device_type);
}

void Training::run(const std::shared_ptr<StateEq> & state_eq,
                   const std::shared_ptr<Dataset> & train_dataset,
                   const std::shared_ptr<Dataset> & test_dataset,
                   const std::string & model_path,
                   int batch_size,
                   int num_epoch)
{
  // Make data loader for train and test
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(*train_dataset), torch::data::DataLoaderOptions(batch_size).workers(2));
  auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(*test_dataset), torch::data::DataLoaderOptions(batch_size).workers(2));
  if(debug_)
  {
    std::cout << "Construct DataLoader" << std::endl;
    std::cout << "  - batch_size: " << batch_size << std::endl;
  }

  // Make model
  auto model_ptr = state_eq->model_ptr_;
  model_ptr->to(*device_);

  // Make optimizer
  torch::optim::Adam optimizer(model_ptr->parameters(), torch::optim::AdamOptions(0.005));

  // Learn
  if(debug_)
  {
    std::cout << "Start training" << std::endl;
    std::cout << "  - device_type: " << device_->type() << std::endl;
  }

  std::ofstream ofs("/tmp/train.dat");
  ofs << "# epoch train_loss test_loss" << std::endl;

  float test_loss_ave_min = std::numeric_limits<float>::max();
  for(int i_epoch = 0; i_epoch < num_epoch; i_epoch++)
  { // For each epoch
    // Train for one epoch
    float train_loss_ave = 0.0;
    for(const std::vector<Example> & batch : *train_data_loader)
    { // For each batch
      // Make batch tensor
      torch::Tensor b_state, b_input, b_next_state_gt;
      makeBatchTensor(batch, *device_, b_state, b_input, b_next_state_gt);

      // Forward, calculate loss, and optimize
      model_ptr->zero_grad();
      torch::Tensor b_next_state_pred = model_ptr->forward(b_state, b_next_state_gt);
      torch::Tensor loss = torch::nn::functional::mse_loss(b_next_state_pred, b_input,
                                                           torch::nn::functional::MSELossFuncOptions(torch::kSum));
      loss.backward();
      optimizer.step();
      train_loss_ave += loss.to(torch::DeviceType::CPU).item<float>();
    }
    train_loss_ave /= static_cast<float>(train_dataset->size().value());

    // Test for one epoch
    float test_loss_ave = 0.0;
    for(const std::vector<Example> & batch : *test_data_loader)
    { // For each batch
      // Not calculate gradient
      torch::NoGradGuard no_grad;

      // Make batch tensor
      torch::Tensor b_state, b_input, b_next_state_gt;
      makeBatchTensor(batch, *device_, b_state, b_input, b_next_state_gt);

      // Forward and calculate loss
      torch::Tensor b_next_state_pred = model_ptr->forward(b_state, b_next_state_gt);
      torch::Tensor loss = torch::nn::functional::mse_loss(b_next_state_pred, b_input,
                                                           torch::nn::functional::MSELossFuncOptions(torch::kSum));
      test_loss_ave += loss.to(torch::DeviceType::CPU).item<float>();
    }
    test_loss_ave /= static_cast<float>(test_dataset->size().value());

    // Print result
    std::cout << "[" << i_epoch << "/" << num_epoch << "] train_loss: " << train_loss_ave
              << ", test_loss: " << test_loss_ave << std::endl;
    ofs << i_epoch << " " << train_loss_ave << " " << test_loss_ave << std::endl;

    // Save model parameters only if the model has the best performance
    if(test_loss_ave < test_loss_ave_min)
    {
      test_loss_ave_min = test_loss_ave;
      std::cout << "Best performance. Save model to " << model_path << std::endl;
      torch::save(model_ptr, model_path);
    }
  }
}
