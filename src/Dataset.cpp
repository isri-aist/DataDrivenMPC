/* Author: Masaki Murooka */

#include <DDMPC/Dataset.h>

using namespace DDMPC;

void DDMPC::makeDataset(const torch::Tensor & state,
                        const torch::Tensor & input,
                        const torch::Tensor & next_state,
                        std::shared_ptr<Dataset> & train_dataset,
                        std::shared_ptr<Dataset> & test_dataset)
{
  // Make dataset for train and test
  int n_all = state.size(0);
  int n_train = 0.7 * n_all;
  int n_test = n_all - n_train;
  torch::Tensor perm = torch::randperm(n_all, torch::kInt64);
  torch::Tensor train_perm = perm.index({at::indexing::Slice(0, n_train)});
  torch::Tensor test_perm = perm.index({at::indexing::Slice(n_train, n_train + n_test)});
  train_dataset =
      std::make_shared<Dataset>(state.index({train_perm}), input.index({train_perm}), next_state.index({train_perm}));
  test_dataset =
      std::make_shared<Dataset>(state.index({test_perm}), input.index({test_perm}), next_state.index({test_perm}));

  // Print debug information
  constexpr bool debug = true;
  if(debug)
  {
    std::cout << "Construct Dataset" << std::endl;
    std::cout << "  - train_dataset size: " << train_dataset->size().value() << std::endl
              << "  - test_dataset size: " << test_dataset->size().value() << std::endl;
  }
}

void DDMPC::makeBatchTensor(const std::vector<Example> & batch,
                            const torch::Device & device,
                            torch::Tensor & b_state,
                            torch::Tensor & b_input,
                            torch::Tensor & b_next_state)
{
  int batch_size = batch.size();

  // Allocate batch tensors
  {
    const auto & data = static_cast<Data>(batch[0]);
    b_state = torch::empty({batch_size, data.state_.size(1)});
    b_input = torch::empty({batch_size, data.input_.size(1)});
    b_next_state = torch::empty({batch_size, data.next_state_.size(1)});
  }

  // Set batch tensors
  for(int i = 0; i < batch_size; i++)
  {
    const auto & data = static_cast<Data>(batch[i]);
    b_state.index({i}) = data.state_;
    b_input.index({i}) = data.input_;
    b_next_state.index({i}) = data.next_state_;
  }

  // Send to device
  b_state = b_state.to(device);
  b_input = b_input.to(device);
  b_next_state = b_next_state.to(device);
}
