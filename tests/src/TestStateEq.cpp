/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/StateEq.h>
#include <DDMPC/TorchUtils.h>

TEST(TestStateEq, Test1)
{
  int state_dim = 3;
  int input_dim = 2;
  DDMPC::StateEq state_eq(state_dim, input_dim);

  Eigen::VectorXd x = Eigen::VectorXd::Random(state_dim);
  Eigen::VectorXd u = Eigen::VectorXd::Random(input_dim);
  Eigen::MatrixXd grad_x(state_dim, state_dim);
  Eigen::MatrixXd grad_u(state_dim, input_dim);

  Eigen::VectorXd next_x = state_eq.eval(x, u, grad_x, grad_u);
  EXPECT_FALSE(next_x.array().isNaN().any());
  EXPECT_FALSE(grad_x.array().isNaN().any());
  EXPECT_FALSE(grad_u.array().isNaN().any());

  Eigen::MatrixXd grad_x_numerical(state_dim, state_dim);
  Eigen::MatrixXd grad_u_numerical(state_dim, input_dim);
  double eps = 1e-4;
  for(int i = 0; i < state_dim; i++)
  {
    grad_x_numerical.col(i) = (state_eq.eval(x + eps * Eigen::VectorXd::Unit(state_dim, i), u)
                               - state_eq.eval(x - eps * Eigen::VectorXd::Unit(state_dim, i), u))
                              / (2 * eps);
  }
  for(int i = 0; i < input_dim; i++)
  {
    grad_u_numerical.col(i) = (state_eq.eval(x, u + eps * Eigen::VectorXd::Unit(input_dim, i))
                               - state_eq.eval(x, u - eps * Eigen::VectorXd::Unit(input_dim, i)))
                              / (2 * eps);
  }
  EXPECT_LT((grad_x - grad_x_numerical).norm(), 1e-3);
  EXPECT_LT((grad_u - grad_u_numerical).norm(), 1e-3);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
