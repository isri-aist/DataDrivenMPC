/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include <nmpc_ddp/DDPSolver.h>

#include <DDMPC/TorchUtils.h>
#include <DDMPC/Training.h>

namespace Eigen
{
using Vector1d = Eigen::Matrix<double, 1, 1>;
}

/** \brief DDP problem based on combination of analytical and data-driven models.

    State consists of [robot_com_pos, robot_com_vel, obj_com_pos, obj_com_vel].
    Input consists of [robot_zmp, obj_force].
 */
class DDPProblem : public nmpc_ddp::DDPProblem<4, 2>
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
             const std::function<void(double, Eigen::Ref<StateDimVector>, Eigen::Ref<InputDimVector>)> & ref_func,
             const WeightParam & weight_param = WeightParam())
  : nmpc_ddp::DDPProblem<4, 2>(dt), state_eq_(state_eq), ref_func_(ref_func), weight_param_(weight_param)
  {
  }

  Eigen::Vector2d simulateRobot(const Eigen::Vector2d & x, const Eigen::Vector2d & u, double dt) const
  {
    double robot_com_pos = x[0];
    double robot_com_vel = x[1];
    double robot_zmp = u[0];
    double obj_force = u[1];

    Eigen::Vector2d x_dot;
    x_dot[0] = robot_com_vel;
    x_dot[1] = gravity_acc_ / robot_com_height_ * (robot_com_pos - robot_zmp)
               - obj_grasp_height_ / (robot_mass_ * robot_com_height_) * obj_force;

    return x + dt * x_dot;
  }

  double damperFunc(double vel) const
  {
    double damper_coeff = 100.0;
    return -1 * damper_coeff * vel;
  }

  Eigen::Vector2d simulateObj(const Eigen::Vector2d & x, const Eigen::Vector1d & u, double dt) const
  {
    double obj_com_pos = x[0];
    double obj_com_vel = x[1];
    double obj_force = u[0];

    Eigen::Vector2d x_dot;
    x_dot[0] = obj_com_vel;
    x_dot[1] = (damperFunc(obj_com_vel) + obj_force) / obj_mass_;

    return x + dt * x_dot;
  }

  StateDimVector simulate(const StateDimVector & x, const InputDimVector & u, double dt) const
  {
    StateDimVector next_x;
    next_x.head<2>() = simulateRobot(x.head<2>(), u, dt);
    next_x.tail<2>() = simulateObj(x.tail<2>(), u.tail<1>(), dt);

    return next_x;
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector next_x;
    next_x.head<2>() = simulateRobot(x.head<2>(), u, dt_);
    next_x.tail<2>() = state_eq_->eval(x.tail<2>(), u.tail<1>());

    return next_x;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector ref_x;
    InputDimVector ref_u;
    ref_func_(t, ref_x, ref_u);

    return 0.5 * weight_param_.running_state.dot((x - ref_x).cwiseAbs2())
           + 0.5 * weight_param_.running_input.dot((u - ref_u).cwiseAbs2());
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    StateDimVector ref_x;
    InputDimVector ref_u;
    ref_func_(t, ref_x, ref_u);

    return 0.5 * weight_param_.terminal_state.dot((x - ref_x).cwiseAbs2());
  }

  virtual void calcStateEqDeriv(double t,
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_deriv_x.setZero();
    state_eq_deriv_u.setZero();

    state_eq_deriv_x(0, 1) = 1;
    state_eq_deriv_x(1, 0) = gravity_acc_ / robot_com_height_;
    state_eq_deriv_x.topRows<2>() *= dt_;
    state_eq_deriv_x.diagonal().head<2>().array() += 1.0;

    state_eq_deriv_u(1, 0) = -1 * gravity_acc_ / robot_com_height_;
    state_eq_deriv_u(1, 1) = -1 * obj_grasp_height_ / (robot_mass_ * robot_com_height_);
    state_eq_deriv_u.topRows<2>() *= dt_;

    state_eq_->eval(x.tail<2>(), u.tail<1>(), state_eq_deriv_x.bottomRightCorner<2, 2>(),
                    state_eq_deriv_u.bottomRightCorner<2, 1>());
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
    StateDimVector ref_x;
    InputDimVector ref_u;
    ref_func_(t, ref_x, ref_u);

    running_cost_deriv_x = weight_param_.running_state.cwiseProduct(x - ref_x);
    running_cost_deriv_u = weight_param_.running_input.cwiseProduct(u - ref_u);
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
    StateDimVector ref_x;
    InputDimVector ref_u;
    ref_func_(t, ref_x, ref_u);

    terminal_cost_deriv_x = weight_param_.terminal_state.cwiseProduct(x - ref_x);
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
  std::shared_ptr<DDMPC::StateEq> state_eq_;

  std::function<void(double, Eigen::Ref<StateDimVector>, Eigen::Ref<InputDimVector>)> ref_func_;

  WeightParam weight_param_;

  double gravity_acc_ = 9.8; // [m/s^2]
  double robot_mass_ = 100.0; // [kg]
  double robot_com_height_ = 1.0; // [m]
  double obj_mass_ = 50.0; // [kg]
  double obj_grasp_height_ = 1.0; // [m]
};

TEST(TestMpcLocomanip, RunMPC)
{
  //// 1. Train state equation ////
  double horizon_dt = 0.1; // [sec]

  // Instantiate state equation
  int obj_state_dim = 2;
  int obj_input_dim = 1;
  int middle_layer_dim = 4;
  auto state_eq = std::make_shared<DDMPC::StateEq>(obj_state_dim, obj_input_dim, middle_layer_dim);

  // Instantiate problem
  auto ref_func = [&](double t, Eigen::Ref<DDPProblem::StateDimVector> ref_x,
                      Eigen::Ref<DDPProblem::InputDimVector> ref_u) {
    // Add small values to avoid numerical instability at inequality bounds
    constexpr double epsilon_t = 1e-6;
    t += epsilon_t;

    // Object position
    ref_x.setZero();
    if(t < 1.0) // [sec]
    {
      ref_x[2] = 0.2; // [m]
    }
    else if(t < 3.0) // [sec]
    {
      ref_x[2] = 0.3 * (t - 1.0) + 0.2; // [m]
    }
    else
    {
      ref_x[2] = 0.8; // [m]
    }

    // ZMP position
    ref_u.setZero();
    double step_total_duration = 1.0; // [sec]
    double step_transit_duration = 0.2; // [sec]
    int step_idx = std::clamp(static_cast<int>(std::floor((t - 1.5) / step_total_duration)), -1, 2);
    if(step_idx == -1)
    {
      ref_u[0] = 0.0; // [m]
    }
    else
    {
      double step_start_time = 1.5 + step_idx * step_total_duration;
      ref_u[0] = 0.2 * (step_idx + std::clamp((t - step_start_time) / step_transit_duration, 0.0, 1.0));
    }
    ref_x[0] = ref_u[0];
  };
  DDPProblem::WeightParam weight_param;
  weight_param.running_state << 0.0, 1e-4, 1e2, 1e-4;
  weight_param.running_input << 1.0, 1e-4;
  weight_param.terminal_state << 1.0, 1.0, 1.0, 1.0;
  auto ddp_problem = std::make_shared<DDPProblem>(horizon_dt, state_eq, ref_func, weight_param);

  // Generate dataset
  auto start_dataset_time = std::chrono::system_clock::now();
  int dataset_size = 20000;
  Eigen::MatrixXd state_all = 1.0 * Eigen::MatrixXd::Random(dataset_size, obj_state_dim);
  Eigen::MatrixXd input_all = 100.0 * Eigen::MatrixXd::Random(dataset_size, obj_input_dim);
  Eigen::MatrixXd next_state_all(dataset_size, obj_state_dim);
  for(int i = 0; i < dataset_size; i++)
  {
    next_state_all.row(i) =
        ddp_problem->simulateObj(state_all.row(i).transpose(), input_all.row(i).transpose(), horizon_dt).transpose();
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
  std::string model_path = "/tmp/TestMpcLocomanipModel.pt";
  int batch_size = 256;
  int num_epoch = 200;
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
  double horizon_duration = 3.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
  double end_t = 5.0; // [sec]

  // Instantiate solver
  auto ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<4, 2>>(ddp_problem);
  auto input_limits_func = [&](double t) -> std::array<DDPProblem::InputDimVector, 2> {
    std::array<DDPProblem::InputDimVector, 2> limits;
    limits[0] << -1.0, -50.0;
    limits[1] << 1.0, 50.0;
    return limits;
  };
  ddp_solver->setInputLimitsFunc(input_limits_func);
  ddp_solver->config().with_input_constraint = true;
  ddp_solver->config().horizon_steps = horizon_steps;

  // Initialize MPC
  double sim_dt = 0.02; // [sec]
  double current_t = 0;
  DDPProblem::StateDimVector current_x = DDPProblem::StateDimVector(0.0, 0.0, 0.2, 0.0);
  std::vector<DDPProblem::InputDimVector> current_u_list(horizon_steps, DDPProblem::InputDimVector::Zero());

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestMpcLocomanipResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time robot_com_pos robot_com_vel obj_com_pos obj_com_vel robot_zmp obj_force ref_obj_com_pos ref_robot_zmp "
         "ddp_iter computation_time"
      << std::endl;
  while(current_t < end_t)
  {
    // Solve
    auto start_time = std::chrono::system_clock::now();
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->config().max_iter = 3;
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
    // TODO
    // EXPECT_LT(std::abs(current_x[0]), 2.0);
    // EXPECT_LT(std::abs(current_x[1]), 2.0);
    // EXPECT_LE(std::abs(current_u[0]), 1.0);

    // Dump
    DDPProblem::StateDimVector current_ref_x;
    DDPProblem::InputDimVector current_ref_u;
    ref_func(current_t, current_ref_x, current_ref_u);
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " " << current_ref_x[2] << " "
        << current_ref_u[0] << " " << ddp_solver->traceDataList().back().iter << " " << duration << std::endl;

    // Update to next step
    current_t += sim_dt;
    current_x = ddp_problem->simulate(current_x, current_u, sim_dt);
    current_u_list = ddp_solver->controlData().u_list;
    current_u_list.erase(current_u_list.begin());
    current_u_list.push_back(current_u_list.back());
  }

  // Final check
  const DDPProblem::InputDimVector & current_u = ddp_solver->controlData().u_list[0];
  // TODO
  // EXPECT_LT(std::abs(current_x[0]), 0.1);
  // EXPECT_LT(std::abs(current_x[1]), 0.5);
  // EXPECT_LT(std::abs(current_u[0]), 0.5);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path
            << "\" u 1:2 w lp, \"\" u 1:4 w lp, \"\" u 1:6 w lp, \"\" u 1:8 w l lw 2, \"\" u 1:9 w l lw 2\n"
            << "  plot \"" << file_path << "\" u 1:7 w lp\n";
}

TEST(TestMpcLocomanip, CheckDerivatives)
{
  constexpr double deriv_eps = 1e-4;

  double horizon_dt = 0.05; // [sec]
  int obj_state_dim = 2;
  int obj_input_dim = 1;
  int middle_layer_dim = 4;
  auto state_eq = std::make_shared<DDMPC::StateEq>(obj_state_dim, obj_input_dim, middle_layer_dim);
  auto ref_func = [&](double t, Eigen::Ref<DDPProblem::StateDimVector> ref_x,
                      Eigen::Ref<DDPProblem::InputDimVector> ref_u) {
    ref_x.setZero();
    ref_u.setZero();
  };
  auto ddp_problem = std::make_shared<DDPProblem>(horizon_dt, state_eq, ref_func);

  double t = 0;
  DDPProblem::StateDimVector x = DDPProblem::StateDimVector::Random();
  DDPProblem::InputDimVector u = DDPProblem::InputDimVector::Random();

  DDPProblem::StateStateDimMatrix state_eq_deriv_x_analytical;
  DDPProblem::StateInputDimMatrix state_eq_deriv_u_analytical;
  ddp_problem->calcStateEqDeriv(t, x, u, state_eq_deriv_x_analytical, state_eq_deriv_u_analytical);

  DDPProblem::StateStateDimMatrix state_eq_deriv_x_numerical;
  DDPProblem::StateInputDimMatrix state_eq_deriv_u_numerical;
  for(int i = 0; i < ddp_problem->stateDim(); i++)
  {
    state_eq_deriv_x_numerical.col(i) =
        (ddp_problem->stateEq(t, x + deriv_eps * DDPProblem::StateDimVector::Unit(i), u)
         - ddp_problem->stateEq(t, x - deriv_eps * DDPProblem::StateDimVector::Unit(i), u))
        / (2 * deriv_eps);
  }
  for(int i = 0; i < ddp_problem->inputDim(); i++)
  {
    state_eq_deriv_u_numerical.col(i) =
        (ddp_problem->stateEq(t, x, u + deriv_eps * DDPProblem::InputDimVector::Unit(i))
         - ddp_problem->stateEq(t, x, u - deriv_eps * DDPProblem::InputDimVector::Unit(i)))
        / (2 * deriv_eps);
  }

  EXPECT_LT((state_eq_deriv_x_analytical - state_eq_deriv_x_numerical).norm(), 1e-3)
      << "state_eq_deriv_x_analytical:\n"
      << state_eq_deriv_x_analytical << std::endl
      << "state_eq_deriv_x_numerical:\n"
      << state_eq_deriv_x_numerical << std::endl
      << "state_eq_deriv_x_error:\n"
      << state_eq_deriv_x_analytical - state_eq_deriv_x_numerical << std::endl;
  EXPECT_LT((state_eq_deriv_u_analytical - state_eq_deriv_u_numerical).norm(), 1e-3)
      << "state_eq_deriv_u_analytical:\n"
      << state_eq_deriv_u_analytical << std::endl
      << "state_eq_deriv_u_numerical:\n"
      << state_eq_deriv_u_numerical << std::endl
      << "state_eq_deriv_u_error:\n"
      << state_eq_deriv_u_analytical - state_eq_deriv_u_numerical << std::endl;
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
