/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include <nmpc_ddp/DDPSolver.h>

#include <DDMPC/TorchUtils.h>
#include <DDMPC/Training.h>

/** \brief DDP problem based on data-driven state equation. */
class DDPProblem : public nmpc_ddp::DDPProblem<2, 1>
{
public:
  struct WeightParam
  {
    StateDimVector running_state;
    InputDimVector running_input;
    StateDimVector terminal_state;

    WeightParam(const StateDimVector & _running_state = StateDimVector::Constant(1.0),
                const InputDimVector & _running_input = InputDimVector::Constant(1.0),
                const StateDimVector & _terminal_state = StateDimVector::Constant(1.0))
    : running_state(_running_state), running_input(_running_input), terminal_state(_terminal_state)
    {
    }
  };

public:
  DDPProblem(double dt,
             const std::shared_ptr<DDMPC::StateEq> & state_eq,
             const WeightParam & weight_param = WeightParam())
  : nmpc_ddp::DDPProblem<2, 1>(dt), state_eq_(state_eq), weight_param_(weight_param)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return state_eq_->eval(x, u);
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
    state_eq_->eval(x, u, state_eq_deriv_x, state_eq_deriv_u);
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

protected:
  WeightParam weight_param_;

  std::shared_ptr<DDMPC::StateEq> state_eq_;
};

namespace Eigen
{
using Vector1d = Eigen::Matrix<double, 1, 1>;
}

// Van der Pol oscillator
// https://web.casadi.org/docs/#a-simple-test-problem
Eigen::Vector2d simulate(const Eigen::Vector2d & x, const Eigen::Vector1d & u, double dt)
{
  Eigen::Vector2d x_dot;
  x_dot[0] = (1.0 - std::pow(x[1], 2)) * x[0] - x[1] + u[0];
  x_dot[1] = x[0];
  return x + dt * x_dot;
}

TEST(TestMPC, Test1)
{
  //// 1. Train state equation ////
  double horizon_dt = 0.03; // [sec]
  int state_dim = 2;
  int input_dim = 1;
  int middle_layer_dim = 16;
  auto state_eq = std::make_shared<DDMPC::StateEq>(state_dim, input_dim, middle_layer_dim);

  // Generate dataset
  auto start_dataset_time = std::chrono::system_clock::now();
  int dataset_size = 100000;
  Eigen::MatrixXd state_all = 2.0 * Eigen::MatrixXd::Random(dataset_size, state_dim);
  Eigen::MatrixXd input_all = 2.0 * Eigen::MatrixXd::Random(dataset_size, input_dim);
  Eigen::MatrixXd next_state_all(dataset_size, state_dim);
  for(int i = 0; i < dataset_size; i++)
  {
    next_state_all.row(i) =
        simulate(state_all.row(i).transpose(), input_all.row(i).transpose(), horizon_dt).transpose();
  }

  // Instantiate dataset
  std::shared_ptr<DDMPC::Dataset> train_dataset;
  std::shared_ptr<DDMPC::Dataset> test_dataset;
  DDMPC::makeDataset(DDMPC::toTorchTensor(state_all.cast<float>()), DDMPC::toTorchTensor(input_all.cast<float>()),
                     DDMPC::toTorchTensor(next_state_all.cast<float>()), train_dataset, test_dataset);
  std::cout << "dataset duration: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()
                                                                         - start_dataset_time)
                   .count()
            << " [s]" << std::endl;

  // Training model
  auto start_train_time = std::chrono::system_clock::now();
  DDMPC::Training training;
  std::string model_path = "/tmp/TestMPCModel.pt";
  int batch_size = 256;
  int num_epoch = 500;
  training.run(state_eq, train_dataset, test_dataset, model_path, batch_size, num_epoch);
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
  double horizon_duration = 5.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
  double end_t = 10.0; // [sec]

  // Instantiate problem
  auto ddp_problem = std::make_shared<DDPProblem>(horizon_dt, state_eq);

  // Instantiate solver
  auto ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<2, 1>>(ddp_problem);
  auto input_limits_func = [&](double t) -> std::array<DDPProblem::InputDimVector, 2> {
    std::array<DDPProblem::InputDimVector, 2> limits;
    limits[0].setConstant(input_dim, -1.0);
    limits[1].setConstant(input_dim, 1.0);
    return limits;
  };
  ddp_solver->setInputLimitsFunc(input_limits_func);
  ddp_solver->config().with_input_constraint = true;
  ddp_solver->config().horizon_steps = horizon_steps;

  // Initialize MPC
  double sim_dt = 0.02; // [sec]
  double current_t = 0;
  DDPProblem::StateDimVector current_x = DDPProblem::StateDimVector(0.0, 1.0);
  std::vector<DDPProblem::InputDimVector> current_u_list(horizon_steps, DDPProblem::InputDimVector::Zero());

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestMPCResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time x[0] x[1] u[0] ddp_iter computation_time" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    auto start_time = std::chrono::system_clock::now();
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->config().max_iter = 5;
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
    EXPECT_LT(std::abs(current_x[0]), 2.0);
    EXPECT_LT(std::abs(current_x[1]), 2.0);
    EXPECT_LE(std::abs(current_u[0]), 1.0);

    // Dump
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " "
        << ddp_solver->traceDataList().back().iter << " " << duration << std::endl;

    // Update to next step
    current_t += sim_dt;
    current_x = simulate(current_x, current_u, sim_dt);
    current_u_list = ddp_solver->controlData().u_list;
    current_u_list.erase(current_u_list.begin());
    current_u_list.push_back(current_u_list.back());
  }

  // Final check
  const DDPProblem::InputDimVector & current_u = ddp_solver->controlData().u_list[0];
  EXPECT_LT(std::abs(current_x[0]), 0.1);
  EXPECT_LT(std::abs(current_x[1]), 0.1);
  EXPECT_LT(std::abs(current_u[0]), 0.1);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:3 w lp, \"\" u 1:4 w lp\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
