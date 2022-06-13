/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/TorchUtils.h>

TEST(TestTorchUtils, Test1)
{
  Eigen::MatrixXf mat1 = Eigen::MatrixXf::Random(4, 10);
  torch::Tensor tensor1 = DDMPC::toTorchTensor(mat1);
  Eigen::MatrixXf mat1_restored = DDMPC::toEigenMatrix(tensor1);
  torch::Tensor tensor1_restored = DDMPC::toTorchTensor(mat1_restored);

  EXPECT_LT((mat1 - mat1_restored).norm(), 1e-8);
  EXPECT_LT((tensor1 - tensor1_restored).norm().item<float>(), 1e-8);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
