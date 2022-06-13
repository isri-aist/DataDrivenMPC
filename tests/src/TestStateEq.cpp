/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

TEST(TestStateEq, Test1)
{
  int state_dim = 2;
  int input_dim = 1;
  DDMPC::StateEq state_eq(state_dim, input_dim);

  Eigen::VectorXd x = Eigen::VectorXd::Random(state_dim);
  Eigen::VectorXd u = Eigen::VectorXd::Random(input_dim);
  Eigen::VectorXd next_x = state_eq.calc(x, u);
  EXPECT_FALSE(next_x.array().isNaN().any());
  EXPECT_GT((next_x - x).norm(), 1e-8);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
