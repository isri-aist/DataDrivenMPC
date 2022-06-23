/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include <nmpc_ddp/DDPSolver.h>

#include <ros/package.h>
#include <ros/ros.h>
#include <data_driven_mpc/Dataset.h>
#include <data_driven_mpc/GenerateDataset.h>
#include <data_driven_mpc/RunSimOnce.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <DDMPC/MathUtils.h>
#include <DDMPC/TorchUtils.h>
#include <DDMPC/Training.h>

/** \brief DDP problem based on data-driven state equation. */
class DDPProblem : public nmpc_ddp::DDPProblem<4, 2>
{
public:
  struct WeightParam
  {
    StateDimVector running_state;
    InputDimVector running_input;
    StateDimVector terminal_state;

    WeightParam(const StateDimVector & _running_state = StateDimVector(1e2, 1e-2, 1e2, 1e-2),
                const InputDimVector & _running_input = InputDimVector::Constant(1e-2),
                const StateDimVector & _terminal_state = StateDimVector(1e2, 1e0, 1e2, 1e0))
    : running_state(_running_state), running_input(_running_input), terminal_state(_terminal_state)
    {
    }
  };

public:
  DDPProblem(double dt,
             const std::shared_ptr<DDMPC::StateEq> & state_eq,
             const WeightParam & weight_param = WeightParam())
  : nmpc_ddp::DDPProblem<4, 2>(dt), state_eq_(state_eq), weight_param_(weight_param)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return next_state_standard_scaler_->applyOneInv(
        state_eq_->eval(state_standard_scaler_->applyOne(x), input_standard_scaler_->applyOne(u)));
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return 0.5 * weight_param_.running_state.dot(x.cwiseAbs2()) + 0.5 * weight_param_.running_input.dot(u.cwiseAbs2());
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    return 0.5 * weight_param_.terminal_state.dot(x.cwiseAbs2());
  }

  virtual void calcStateEqDeriv(double t,
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_->eval(state_standard_scaler_->applyOne(x), input_standard_scaler_->applyOne(u), state_eq_deriv_x,
                    state_eq_deriv_u);
    state_eq_deriv_x.array().colwise() *= next_state_standard_scaler_->stddev_vec_.transpose().array();
    state_eq_deriv_x.array().rowwise() /= state_standard_scaler_->stddev_vec_.array();
    state_eq_deriv_u.array().colwise() *= next_state_standard_scaler_->stddev_vec_.transpose().array();
    state_eq_deriv_u.array().rowwise() /= input_standard_scaler_->stddev_vec_.array();
  }

  virtual void calcStateEqDeriv(double t,
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                                std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                                std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                                std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const override
  {
    throw std::runtime_error("Second-order derivatives of state equation are not implemented.");
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    running_cost_deriv_x = weight_param_.running_state.cwiseProduct(x);
    running_cost_deriv_u = weight_param_.running_input.cwiseProduct(u);
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u,
                                    Eigen::Ref<StateStateDimMatrix> running_cost_deriv_xx,
                                    Eigen::Ref<InputInputDimMatrix> running_cost_deriv_uu,
                                    Eigen::Ref<StateInputDimMatrix> running_cost_deriv_xu) const override
  {
    calcRunningCostDeriv(t, x, u, running_cost_deriv_x, running_cost_deriv_u);

    running_cost_deriv_xx.setZero();
    running_cost_deriv_xx.diagonal() = weight_param_.running_state;
    running_cost_deriv_uu.setZero();
    running_cost_deriv_uu.diagonal() = weight_param_.running_input;
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    terminal_cost_deriv_x = weight_param_.terminal_state.cwiseProduct(x);
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    calcTerminalCostDeriv(t, x, terminal_cost_deriv_x);

    terminal_cost_deriv_xx.setZero();
    terminal_cost_deriv_xx.diagonal() = weight_param_.terminal_state;
  }

  void setStandardScaler(const std::shared_ptr<DDMPC::StandardScaler<double, 4>> & state_standard_scaler,
                         const std::shared_ptr<DDMPC::StandardScaler<double, 2>> & input_standard_scaler,
                         const std::shared_ptr<DDMPC::StandardScaler<double, 4>> & next_state_standard_scaler)

  {
    state_standard_scaler_ = state_standard_scaler;
    input_standard_scaler_ = input_standard_scaler;
    next_state_standard_scaler_ = next_state_standard_scaler;
  }

protected:
  std::shared_ptr<DDMPC::StateEq> state_eq_;

  WeightParam weight_param_;

  std::shared_ptr<DDMPC::StandardScaler<double, 4>> state_standard_scaler_;
  std::shared_ptr<DDMPC::StandardScaler<double, 2>> input_standard_scaler_;
  std::shared_ptr<DDMPC::StandardScaler<double, 4>> next_state_standard_scaler_;
};

namespace Eigen
{
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

TEST(TestMpcCart, Test1)
{
  ros::NodeHandle nh;
  ros::ServiceClient generate_dataset_cli = nh.serviceClient<data_driven_mpc::GenerateDataset>("/generate_dataset");
  ros::ServiceClient run_sim_once_cli = nh.serviceClient<data_driven_mpc::RunSimOnce>("/run_sim_once");
  ASSERT_TRUE(generate_dataset_cli.waitForExistence(ros::Duration(10.0)))
      << "[TestMpcCart] Failed to wait for ROS service to generate dataset." << std::endl;
  ASSERT_TRUE(run_sim_once_cli.waitForExistence(ros::Duration(10.0)))
      << "[TestMpcCart] Failed to wait for ROS service to run simulation once." << std::endl;

  //// 1. Train state equation ////
  double horizon_dt = 0.1; // [sec]

  // Instantiate state equation
  int state_dim = 4;
  int input_dim = 2;
  int middle_layer_dim = 64;
  auto state_eq = std::make_shared<DDMPC::StateEq>(state_dim, input_dim, middle_layer_dim);

  // Instantiate problem
  auto ddp_problem = std::make_shared<DDPProblem>(horizon_dt, state_eq);

  // Call service to generate dataset
  auto start_dataset_time = std::chrono::system_clock::now();
  data_driven_mpc::GenerateDataset generate_dataset_srv;
  std::string dataset_filename = ros::package::getPath("data_driven_mpc") + "/tests/data/TestMpcCartDataset.bag";
  int dataset_size = 100000;
  DDPProblem::StateDimVector x_max = DDPProblem::StateDimVector(1.0, 0.2, 0.4, 0.5);
  DDPProblem::InputDimVector u_max = DDPProblem::InputDimVector(20.0, 20.0);
  generate_dataset_srv.request.filename = dataset_filename;
  generate_dataset_srv.request.dataset_size = dataset_size;
  generate_dataset_srv.request.dt = horizon_dt;
  generate_dataset_srv.request.state_max.resize(state_dim);
  DDPProblem::StateDimVector::Map(&generate_dataset_srv.request.state_max[0], state_dim) = x_max;
  generate_dataset_srv.request.state_min.resize(state_dim);
  DDPProblem::StateDimVector::Map(&generate_dataset_srv.request.state_min[0], state_dim) = -1 * x_max;
  generate_dataset_srv.request.input_max.resize(input_dim);
  DDPProblem::InputDimVector::Map(&generate_dataset_srv.request.input_max[0], input_dim) = u_max;
  generate_dataset_srv.request.input_min.resize(input_dim);
  DDPProblem::InputDimVector::Map(&generate_dataset_srv.request.input_min[0], input_dim) = -1 * u_max;
  ASSERT_TRUE(generate_dataset_cli.call(generate_dataset_srv))
      << "[TestMpcCart] Failed to call ROS service to generate dataset." << std::endl;

  // Load dataset from rosbag
  Eigen::MatrixXd state_all;
  Eigen::MatrixXd input_all;
  Eigen::MatrixXd next_state_all;
  rosbag::Bag dataset_bag;
  dataset_bag.open(dataset_filename, rosbag::bagmode::Read);
  for(rosbag::MessageInstance const msg : rosbag::View(dataset_bag, rosbag::TopicQuery({"/dataset"})))
  {
    data_driven_mpc::Dataset::ConstPtr dataset_msg = msg.instantiate<data_driven_mpc::Dataset>();
    state_all = Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->state_all.data(), dataset_size, state_dim);
    input_all = Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->input_all.data(), dataset_size, input_dim);
    next_state_all =
        Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->next_state_all.data(), dataset_size, state_dim);
    break;
  }
  dataset_bag.close();

  // Instantiate standardization scalar
  auto state_standard_scaler = std::make_shared<DDMPC::StandardScaler<double, 4>>(state_all);
  auto input_standard_scaler = std::make_shared<DDMPC::StandardScaler<double, 2>>(input_all);
  auto next_state_standard_scaler = std::make_shared<DDMPC::StandardScaler<double, 4>>(next_state_all);
  ddp_problem->setStandardScaler(state_standard_scaler, input_standard_scaler, next_state_standard_scaler);

  // Instantiate dataset
  std::shared_ptr<DDMPC::Dataset> train_dataset;
  std::shared_ptr<DDMPC::Dataset> test_dataset;
  DDMPC::makeDataset(DDMPC::toTorchTensor(state_standard_scaler->apply(state_all).cast<float>()),
                     DDMPC::toTorchTensor(input_standard_scaler->apply(input_all).cast<float>()),
                     DDMPC::toTorchTensor(next_state_standard_scaler->apply(next_state_all).cast<float>()),
                     train_dataset, test_dataset);
  std::cout << "dataset duration: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()
                                                                         - start_dataset_time)
                   .count()
            << " [s]" << std::endl;

  // Training model
  auto start_train_time = std::chrono::system_clock::now();
  DDMPC::Training training;
  std::string model_path = ros::package::getPath("data_driven_mpc") + "/tests/data/TestMpcCartModel.pt";
  int batch_size = 256;
  int num_epoch = 200;
  double learning_rate = 1e-3;
  training.run(state_eq, train_dataset, test_dataset, model_path, batch_size, num_epoch, learning_rate);
  training.load(state_eq, model_path);
  std::cout << "train duration: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()
                                                                         - start_train_time)
                   .count()
            << " [s]" << std::endl;

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"/tmp/DataDrivenMPCTraining.txt\" u 1:2 w lp, \"\" u 1:3 w lp\n";

  //// 2. Run MPC ////
  double horizon_duration = 2.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
  double end_t = 5.0; // [sec]

  // Instantiate solver
  auto ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<4, 2>>(ddp_problem);
  auto input_limits_func = [&](double t) -> std::array<DDPProblem::InputDimVector, 2> {
    std::array<DDPProblem::InputDimVector, 2> limits;
    limits[0] = -1 * u_max;
    limits[1] = u_max;
    return limits;
  };
  ddp_solver->setInputLimitsFunc(input_limits_func);
  ddp_solver->config().with_input_constraint = true;
  ddp_solver->config().horizon_steps = horizon_steps;
  ddp_solver->config().max_iter = 3;

  // Initialize MPC
  double sim_dt = 0.05; // [sec]
  double current_t = 0;
  DDPProblem::StateDimVector current_x = DDPProblem::StateDimVector(-0.2, 0.0, 0.2, 0.0);
  std::vector<DDPProblem::InputDimVector> current_u_list(horizon_steps, DDPProblem::InputDimVector::Zero());

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestMpcCartResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time p p_dot theta theta_dot fx fz ddp_iter computation_time" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    auto start_time = std::chrono::system_clock::now();
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
    }

    // Set input
    const auto & input_limits = input_limits_func(current_t);
    DDPProblem::InputDimVector current_u =
        ddp_solver->controlData().u_list[0].cwiseMax(input_limits[0]).cwiseMin(input_limits[1]);
    double duration =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();

    // Check
    for(int i = 0; i < state_dim; i++)
    {
      EXPECT_LE(std::abs(current_x[i]), 2 * x_max[i]) << "[TestMpcCart] Violate x[" << i << "] limits." << std::endl;
    }
    for(int i = 0; i < input_dim; i++)
    {
      EXPECT_LE(std::abs(current_u[i]), 2 * u_max[i]) << "[TestMpcCart] Violate u[" << i << "] limits." << std::endl;
    }

    // Dump
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " "
        << ddp_solver->traceDataList().back().iter << " " << duration << std::endl;

    // Update to next step
    current_t += sim_dt;
    data_driven_mpc::RunSimOnce run_sim_once_srv;
    run_sim_once_srv.request.dt = sim_dt;
    run_sim_once_srv.request.state.resize(state_dim);
    DDPProblem::StateDimVector::Map(&run_sim_once_srv.request.state[0], state_dim) = current_x;
    run_sim_once_srv.request.input.resize(input_dim);
    DDPProblem::InputDimVector::Map(&run_sim_once_srv.request.input[0], input_dim) = current_u;
    ASSERT_TRUE(run_sim_once_cli.call(run_sim_once_srv))
        << "[TestMpcCart] Failed to call ROS service to run simulation once." << std::endl;
    current_x = DDPProblem::StateDimVector::Map(&run_sim_once_srv.response.state[0], state_dim);
    current_u_list = ddp_solver->controlData().u_list;
  }

  // Final check
  const DDPProblem::InputDimVector & current_u = ddp_solver->controlData().u_list[0];
  EXPECT_LT(std::abs(current_x[0]), 0.1);
  EXPECT_LT(std::abs(current_x[1]), 0.1);
  EXPECT_LT(std::abs(current_x[2]), 0.1);
  EXPECT_LT(std::abs(current_x[3]), 0.1);
  EXPECT_LT(std::abs(current_u[0]), 10.0);
  EXPECT_LT(std::abs(current_u[1]), 20.0);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:4 w lp # state\n"
            << "  plot \"" << file_path << "\" u 1:6 w lp, \"\" u 1:7 w lp # input\n";
}

int main(int argc, char ** argv)
{
  // Setup ROS
  ros::init(argc, argv, "test_mpc_cart");

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
