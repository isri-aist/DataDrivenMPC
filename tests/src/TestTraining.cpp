/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/TorchUtils.h>
#include <DDMPC/Training.h>

TEST(TestTraining, Test1)
{
  // Generate dataset
  int state_dim = 2;
  int input_dim = 1;
  int dataset_size = 1000;
  Eigen::MatrixXd state_all = Eigen::MatrixXd::Random(dataset_size, state_dim);
  Eigen::MatrixXd input_all = Eigen::MatrixXd::Random(dataset_size, input_dim);
  Eigen::MatrixXd next_state_all(dataset_size, state_dim);
  next_state_all.col(0) = 1.0 * state_all.col(0) + 2.0 * state_all.col(1) - 1.0 * input_all.col(0);
  next_state_all.col(1) = -2.0 * state_all.col(0) + -1.0 * state_all.col(1) + 2.0 * input_all.col(0);

  // Instantiate state equation and dataset
  auto state_eq = std::make_shared<DDMPC::StateEq>(state_dim, input_dim);
  std::shared_ptr<DDMPC::Dataset> train_dataset;
  std::shared_ptr<DDMPC::Dataset> test_dataset;
  DDMPC::makeDataset(DDMPC::toTorchTensor(state_all.cast<float>()), DDMPC::toTorchTensor(input_all.cast<float>()),
                     DDMPC::toTorchTensor(next_state_all.cast<float>()), train_dataset, test_dataset);

  double before_error = (next_state_all.row(0).transpose() - state_eq->eval(state_all.row(0), input_all.row(0))).norm();

  // Training model
  DDMPC::Training training;
  std::string model_path = "/tmp/TestTrainingModel.pt";
  training.run(state_eq, train_dataset, test_dataset, model_path);

  double after_error = (next_state_all.row(0).transpose() - state_eq->eval(state_all.row(0), input_all.row(0))).norm();

  // Check error
  EXPECT_LT(after_error, before_error);
  EXPECT_LT(after_error, 0.2);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"/tmp/DataDrivenMPCTraining.txt\" u 1:2 w lp, \"\" u 1:3 w lp\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
