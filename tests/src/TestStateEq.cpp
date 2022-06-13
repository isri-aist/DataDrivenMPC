/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

TEST(TestStateEq, Test1)
{
  constexpr int state_dim = 2;
  constexpr int input_dim = 1;
  DDMPC::StateEq<state_dim, input_dim> state_eq;

  Eigen::Matrix<double, state_dim, 1> x = Eigen::Matrix<double, state_dim, 1>::Random();
  Eigen::Matrix<double, input_dim, 1> u = Eigen::Matrix<double, input_dim, 1>::Random();
  Eigen::Matrix<double, state_dim, 1> next_x = state_eq.calc(x, u);
  EXPECT_GT((next_x - x).norm(), 1e-8);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
