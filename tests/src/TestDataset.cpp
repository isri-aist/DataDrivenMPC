/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/Dataset.h>
#include <DDMPC/TorchUtils.h>

TEST(TestDataset, Test1)
{
  int dataset_size = 100;
  int state_dim = 2;
  int input_dim = 1;
  torch::Tensor state_all = DDMPC::toTorchTensor(Eigen::MatrixXf::Random(dataset_size, state_dim));
  torch::Tensor input_all = DDMPC::toTorchTensor(Eigen::MatrixXf::Random(dataset_size, input_dim));
  torch::Tensor next_state_all = DDMPC::toTorchTensor(Eigen::MatrixXf::Random(dataset_size, state_dim));

  DDMPC::Dataset dataset(state_all, input_all, next_state_all);

  for(int i = 0; i < dataset_size; i++)
  {
    const DDMPC::Data & data = static_cast<DDMPC::Data>(dataset.get(i));
    EXPECT_LT((data.state_ - state_all[i]).norm().item<float>(), 1e-8);
    EXPECT_LT((data.input_ - input_all[i]).norm().item<float>(), 1e-8);
    EXPECT_LT((data.next_state_ - next_state_all[i]).norm().item<float>(), 1e-8);
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
