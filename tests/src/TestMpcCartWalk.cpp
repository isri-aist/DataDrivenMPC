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

/** \brief DDP problem based on combination of analytical and data-driven models.

    State consists of [robot_com_pos_x, robot_com_vel_x, robot_com_pos_y, robot_com_vel_y, obj_p, obj_p_dot, obj_theta,
   obj_theta_dot]. Input consists of [robot_zmp_x, robot_zmp_y, obj_fx, obj_fz]. obj_p is the cart position. obj_theta
   is the cart angle. obj_fx and obj_fz are the manipulation force applied to an object in the X and Z directions.
 */
class DDPProblem : public nmpc_ddp::DDPProblem<8, 4>
{
public:
  static constexpr int RobotStateDim = 4;
  static constexpr int ObjStateDim = 4;
  static constexpr int ObjInputDim = 2;

public:
  using RobotStateDimVector = Eigen::Matrix<double, RobotStateDim, 1>;
  using ObjStateDimVector = Eigen::Matrix<double, ObjStateDim, 1>;
  using ObjInputDimVector = Eigen::Matrix<double, ObjInputDim, 1>;

public:
  struct WeightParam
  {
    StateDimVector running_state;
    InputDimVector running_input;
    StateDimVector terminal_state;

    WeightParam()
    {
      running_state << 0.0, 1e-4, 0.0, 1e-4, 1e2, 1e-4, 1e2, 1e-4;
      running_input << 1.0, 1.0, 1e-2, 1e-2;
      terminal_state << 1.0, 1.0, 1.0, 1.0, 1e2, 1.0, 1.0, 1.0;
    }
  };

public:
  DDPProblem(
      double dt,
      const std::shared_ptr<DDMPC::StateEq> & state_eq,
      const std::function<void(double, Eigen::Ref<StateDimVector>, Eigen::Ref<InputDimVector>)> & ref_func = nullptr,
      const WeightParam & weight_param = WeightParam())
  : nmpc_ddp::DDPProblem<8, 4>(dt), state_eq_(state_eq), ref_func_(ref_func), weight_param_(weight_param)
  {
    _setupRunSimOnce();
  }

  RobotStateDimVector simulateRobot(const StateDimVector & x, const InputDimVector & u, double dt) const
  {
    double robot_com_pos_x = x[0];
    double robot_com_vel_x = x[1];
    double robot_com_pos_y = x[2];
    double robot_com_vel_y = x[3];
    double obj_p = x[4];
    double robot_zmp_x = u[0];
    double robot_zmp_y = u[1];
    double obj_fx = u[2];
    double obj_fz = u[3];

    // See equation (3) in https://hal.archives-ouvertes.fr/hal-03425667/
    double omega2 = gravity_acc_ / robot_com_height_;
    double zeta = robot_mass_ * gravity_acc_;
    double kappa = 1.0 + obj_fz / zeta;
    double gamma_x = (-1 * grasp_pos_z_ * obj_fx + (obj_p + grasp_pos_x_local_) * obj_fz) / zeta;

    RobotStateDimVector x_dot;
    x_dot[0] = robot_com_vel_x;
    x_dot[1] = omega2 * (robot_com_pos_x - kappa * robot_zmp_x + gamma_x);
    x_dot[2] = robot_com_vel_y;
    x_dot[3] = omega2 * (robot_com_pos_y - kappa * robot_zmp_y);

    return x.head<4>() + dt * x_dot;
  }

  ObjStateDimVector simulateObj(const ObjStateDimVector & x, const ObjInputDimVector & u, double dt) const
  {
    data_driven_mpc::RunSimOnce run_sim_once_srv;
    run_sim_once_srv.request.dt = dt;
    run_sim_once_srv.request.state.resize(ObjStateDim);
    ObjStateDimVector::Map(&run_sim_once_srv.request.state[0], ObjStateDim) = x;
    run_sim_once_srv.request.input.resize(ObjInputDim);
    ObjInputDimVector::Map(&run_sim_once_srv.request.input[0], ObjInputDim) = u;
    _callRunSimOnce(run_sim_once_srv);
    return ObjStateDimVector::Map(&run_sim_once_srv.response.state[0], ObjStateDim);
  }

  StateDimVector simulate(const StateDimVector & x, const InputDimVector & u, double dt) const
  {
    StateDimVector next_x;
    next_x.head<4>() = simulateRobot(x, u, dt);
    next_x.tail<4>() = simulateObj(x.tail<4>(), u.tail<2>(), dt);

    return next_x;
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector next_x;
    next_x.head<4>() = simulateRobot(x, u, dt_);
    next_x.tail<4>() = next_state_standard_scaler_->applyOneInv(
        state_eq_->eval(state_standard_scaler_->applyOne(x.tail<4>()), input_standard_scaler_->applyOne(u.tail<2>())));

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
    double robot_com_pos_x = x[0];
    double robot_com_vel_x = x[1];
    double robot_com_pos_y = x[2];
    double robot_com_vel_y = x[3];
    double obj_p = x[4];
    double robot_zmp_x = u[0];
    double robot_zmp_y = u[1];
    double obj_fx = u[2];
    double obj_fz = u[3];

    // See equation (3) in https://hal.archives-ouvertes.fr/hal-03425667/
    double omega2 = gravity_acc_ / robot_com_height_;
    double zeta = robot_mass_ * gravity_acc_;
    double kappa = 1.0 + obj_fz / zeta;
    double gamma_x = (-1 * grasp_pos_z_ * obj_fx + (obj_p + grasp_pos_x_local_) * obj_fz) / zeta;

    state_eq_deriv_x.setZero();
    state_eq_deriv_u.setZero();

    state_eq_deriv_x(0, 1) = 1;
    state_eq_deriv_x(1, 0) = omega2;
    state_eq_deriv_x(1, 4) = omega2 * obj_fz / zeta;
    state_eq_deriv_x(2, 3) = 1;
    state_eq_deriv_x(3, 2) = omega2;
    state_eq_deriv_x.topRows<4>() *= dt_;
    state_eq_deriv_x.diagonal().head<4>().array() += 1.0;

    state_eq_deriv_u(1, 0) = -1 * omega2 * kappa;
    state_eq_deriv_u(1, 2) = -1 * omega2 * grasp_pos_z_ / zeta;
    state_eq_deriv_u(1, 3) = (omega2 / zeta) * (-1 * robot_zmp_x + (obj_p + grasp_pos_x_local_));
    state_eq_deriv_u(3, 1) = -1 * omega2 * kappa;
    state_eq_deriv_u(3, 3) = -1 * omega2 * robot_zmp_y / zeta;
    state_eq_deriv_u.topRows<4>() *= dt_;

    state_eq_->eval(state_standard_scaler_->applyOne(x.tail<4>()), input_standard_scaler_->applyOne(u.tail<2>()),
                    state_eq_deriv_x.bottomRightCorner<4, 4>(), state_eq_deriv_u.bottomRightCorner<4, 2>());
    state_eq_deriv_x.bottomRightCorner<4, 4>().array().colwise() *=
        next_state_standard_scaler_->stddev_vec_.transpose().array();
    state_eq_deriv_x.bottomRightCorner<4, 4>().array().rowwise() /= state_standard_scaler_->stddev_vec_.array();
    state_eq_deriv_u.bottomRightCorner<4, 2>().array().colwise() *=
        next_state_standard_scaler_->stddev_vec_.transpose().array();
    state_eq_deriv_u.bottomRightCorner<4, 2>().array().rowwise() /= input_standard_scaler_->stddev_vec_.array();
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

  void setStandardScaler(const std::shared_ptr<DDMPC::StandardScaler<double, ObjStateDim>> & state_standard_scaler,
                         const std::shared_ptr<DDMPC::StandardScaler<double, ObjInputDim>> & input_standard_scaler,
                         const std::shared_ptr<DDMPC::StandardScaler<double, ObjStateDim>> & next_state_standard_scaler)

  {
    state_standard_scaler_ = state_standard_scaler;
    input_standard_scaler_ = input_standard_scaler;
    next_state_standard_scaler_ = next_state_standard_scaler;
  }

protected:
  void _setupRunSimOnce() const
  {
    // This is a wrapper for google-test's ASSERT_*, which can only be used with void functions
    // https://google.github.io/googletest/advanced.html#assertion-placement
    ASSERT_TRUE(ros::service::waitForService("/run_sim_once", ros::Duration(10.0)))
        << "[TestMpcCart] Failed to wait for ROS service to run simulation once." << std::endl;
  }

  void _callRunSimOnce(data_driven_mpc::RunSimOnce & run_sim_once_srv) const
  {
    // This is a wrapper for google-test's ASSERT_*, which can only be used with void functions
    // https://google.github.io/googletest/advanced.html#assertion-placement
    ASSERT_TRUE(ros::service::call("/run_sim_once", run_sim_once_srv))
        << "[TestMpcCartWalk] Failed to call ROS service to run simulation once." << std::endl;
  }

protected:
  std::shared_ptr<DDMPC::StateEq> state_eq_;

  std::function<void(double, Eigen::Ref<StateDimVector>, Eigen::Ref<InputDimVector>)> ref_func_;

  WeightParam weight_param_;

  std::shared_ptr<DDMPC::StandardScaler<double, ObjStateDim>> state_standard_scaler_;
  std::shared_ptr<DDMPC::StandardScaler<double, ObjInputDim>> input_standard_scaler_;
  std::shared_ptr<DDMPC::StandardScaler<double, ObjStateDim>> next_state_standard_scaler_;

  double gravity_acc_ = 9.8; // [m/s^2]
  double robot_mass_ = 60.0; // [kg]
  double robot_com_height_ = 0.8; // [m]
  double grasp_pos_x_local_ = -0.35; // [m]
  double grasp_pos_z_ = 0.5; // [m]
};

namespace Eigen
{
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

TEST(TestMpcCartWalk, Test1)
{
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::ServiceClient generate_dataset_cli = nh.serviceClient<data_driven_mpc::GenerateDataset>("/generate_dataset");
  ASSERT_TRUE(generate_dataset_cli.waitForExistence(ros::Duration(10.0)))
      << "[TestMpcCartWalk] Failed to wait for ROS service to generate dataset." << std::endl;

  //// 1. Train state equation ////
  double horizon_dt = 0.1; // [sec]

  // Instantiate state equation
  int middle_layer_dim = 32;
  auto state_eq = std::make_shared<DDMPC::StateEq>(DDPProblem::ObjStateDim, DDPProblem::ObjInputDim, middle_layer_dim);

  // Instantiate problem
  auto ref_func = [&](double t, Eigen::Ref<DDPProblem::StateDimVector> ref_x,
                      Eigen::Ref<DDPProblem::InputDimVector> ref_u) -> void {
    // Add small values to avoid numerical instability at inequality bounds
    constexpr double epsilon_t = 1e-6;
    t += epsilon_t;

    // Object position
    ref_x.setZero();
    if(t < 1.0) // [sec]
    {
      ref_x[4] = 0.5; // [m]
    }
    else if(t < 3.0) // [sec]
    {
      ref_x[4] = 0.3 * (t - 1.0) + 0.5; // [m]
    }
    else
    {
      ref_x[4] = 1.1; // [m]
    }

    // ZMP position
    ref_u.setZero();
    double step_total_duration = 1.0; // [sec]
    double step_transit_duration = 0.2; // [sec]
    double zmp_x_step = 0.2; // [m]
    std::vector<double> zmp_y_list = {0.0, -0.1, 0.1, 0.0}; // [m]
    int step_idx = std::clamp(static_cast<int>(std::floor((t - 1.5) / step_total_duration)), -1, 2);
    if(step_idx == -1)
    {
      ref_u[0] = 0.0; // [m]
      ref_u[1] = 0.0; // [m]
    }
    else
    {
      double step_start_time = 1.5 + step_idx * step_total_duration;
      double ratio = std::clamp((t - step_start_time) / step_transit_duration, 0.0, 1.0);
      ref_u[0] = zmp_x_step * (static_cast<double>(step_idx) + ratio);
      ref_u[1] = (1.0 - ratio) * zmp_y_list[step_idx] + ratio * zmp_y_list[step_idx + 1];
    }
    ref_x[0] = ref_u[0];
    ref_x[1] = ref_u[1];
  };
  auto ddp_problem = std::make_shared<DDPProblem>(horizon_dt, state_eq, ref_func);

  // Call service to generate dataset
  auto start_dataset_time = std::chrono::system_clock::now();
  data_driven_mpc::GenerateDataset generate_dataset_srv;
  std::string dataset_filename = ros::package::getPath("data_driven_mpc") + "/tests/data/TestMpcCartWalkDataset.bag";
  int dataset_size = 10000;
  DDPProblem::ObjStateDimVector x_max = DDPProblem::ObjStateDimVector(2.0, 0.2, 0.4, 0.5);
  DDPProblem::ObjStateDimVector x_min = DDPProblem::ObjStateDimVector(0.0, -0.2, -0.4, -0.5);
  DDPProblem::ObjInputDimVector u_max = DDPProblem::ObjInputDimVector(15.0, 15.0);
  generate_dataset_srv.request.filename = dataset_filename;
  generate_dataset_srv.request.dataset_size = dataset_size;
  generate_dataset_srv.request.dt = horizon_dt;
  generate_dataset_srv.request.state_max.resize(DDPProblem::ObjStateDim);
  DDPProblem::ObjStateDimVector::Map(&generate_dataset_srv.request.state_max[0], DDPProblem::ObjStateDim) = x_max;
  generate_dataset_srv.request.state_min.resize(DDPProblem::ObjStateDim);
  DDPProblem::ObjStateDimVector::Map(&generate_dataset_srv.request.state_min[0], DDPProblem::ObjStateDim) = x_min;
  generate_dataset_srv.request.input_max.resize(DDPProblem::ObjInputDim);
  DDPProblem::ObjInputDimVector::Map(&generate_dataset_srv.request.input_max[0], DDPProblem::ObjInputDim) = u_max;
  generate_dataset_srv.request.input_min.resize(DDPProblem::ObjInputDim);
  DDPProblem::ObjInputDimVector::Map(&generate_dataset_srv.request.input_min[0], DDPProblem::ObjInputDim) = -1 * u_max;
  ASSERT_TRUE(generate_dataset_cli.call(generate_dataset_srv))
      << "[TestMpcCartWalk] Failed to call ROS service to generate dataset." << std::endl;

  // Load dataset from rosbag
  Eigen::MatrixXd state_all;
  Eigen::MatrixXd input_all;
  Eigen::MatrixXd next_state_all;
  rosbag::Bag dataset_bag;
  dataset_bag.open(dataset_filename, rosbag::bagmode::Read);
  for(rosbag::MessageInstance const msg : rosbag::View(dataset_bag, rosbag::TopicQuery({"/dataset"})))
  {
    data_driven_mpc::Dataset::ConstPtr dataset_msg = msg.instantiate<data_driven_mpc::Dataset>();
    state_all =
        Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->state_all.data(), dataset_size, DDPProblem::ObjStateDim);
    input_all =
        Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->input_all.data(), dataset_size, DDPProblem::ObjInputDim);
    next_state_all = Eigen::Map<const Eigen::MatrixXdRowMajor>(dataset_msg->next_state_all.data(), dataset_size,
                                                               DDPProblem::ObjStateDim);
    break;
  }
  dataset_bag.close();

  // Instantiate standardization scalar
  auto state_standard_scaler = std::make_shared<DDMPC::StandardScaler<double, DDPProblem::ObjStateDim>>(state_all);
  auto input_standard_scaler = std::make_shared<DDMPC::StandardScaler<double, DDPProblem::ObjInputDim>>(input_all);
  auto next_state_standard_scaler =
      std::make_shared<DDMPC::StandardScaler<double, DDPProblem::ObjStateDim>>(next_state_all);
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
  std::string model_path = ros::package::getPath("data_driven_mpc") + "/tests/data/TestMpcCartWalkModel.pt";
  int batch_size = 256;
  int num_epoch = 250;
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
  auto ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<8, 4>>(ddp_problem);
  auto input_limits_func = [&](double t) -> std::array<DDPProblem::InputDimVector, 2> {
    std::array<DDPProblem::InputDimVector, 2> limits;
    limits[0] << Eigen::Vector2d::Constant(-1e10), -1 * u_max;
    limits[1] << Eigen::Vector2d::Constant(1e10), u_max;
    return limits;
  };
  ddp_solver->setInputLimitsFunc(input_limits_func);
  ddp_solver->config().with_input_constraint = true;
  ddp_solver->config().horizon_steps = horizon_steps;
  ddp_solver->config().max_iter = 5;

  // Initialize MPC
  double sim_dt = 0.05; // [sec]
  double current_t = 0;
  DDPProblem::StateDimVector current_x;
  current_x << 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0;
  std::vector<DDPProblem::InputDimVector> current_u_list(horizon_steps, DDPProblem::InputDimVector::Zero());

  // Run MPC loop
  std::string file_path = "/tmp/TestMpcCartWalkResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time robot_com_pos_x robot_com_vel_x robot_com_pos_y robot_com_vel_y obj_p obj_p_dot obj_theta obj_theta_dot "
         "robot_zmp_x robot_zmp_y obj_fx obj_fz ref_robot_com_pos_x ref_robot_com_vel_x ref_robot_com_pos_y "
         "ref_robot_com_vel_y ref_obj_p ref_obj_p_dot ref_obj_theta ref_obj_theta_dot ref_robot_zmp_x ref_robot_zmp_y "
         "ref_obj_fx ref_obj_fz ddp_iter computation_time"
      << std::endl;
  bool no_exit = false;
  pnh.getParam("no_exit", no_exit);
  while(no_exit || current_t < end_t)
  {
    // Solve
    auto start_time = std::chrono::system_clock::now();
    ddp_solver->solve(current_t, current_x, current_u_list);

    // Set input
    const auto & input_limits = input_limits_func(current_t);
    DDPProblem::InputDimVector current_u =
        ddp_solver->controlData().u_list[0].cwiseMax(input_limits[0]).cwiseMin(input_limits[1]);
    double duration =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();

    // Check
    DDPProblem::StateDimVector current_ref_x;
    DDPProblem::InputDimVector current_ref_u;
    ref_func(current_t, current_ref_x, current_ref_u);
    for(int i = 0; i < current_x.size(); i++)
    {
      EXPECT_LT(std::abs(current_x[i] - current_ref_x[i]), 10.0)
          << "[TestMpcCartWalk] Violate running check for x[" << i << "]." << std::endl;
    }
    for(int i = 0; i < current_u.size(); i++)
    {
      EXPECT_LT(std::abs(current_u[i] - current_ref_u[i]), 100.0)
          << "[TestMpcCartWalk] Violate running check for u[" << i << "]." << std::endl;
    }
    EXPECT_LE(std::abs(current_u[2]), u_max[0]); // [N]
    EXPECT_LE(std::abs(current_u[3]), u_max[1]); // [N]

    // Dump
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " " << current_ref_x.transpose()
        << " " << current_ref_u.transpose() << " " << ddp_solver->traceDataList().back().iter << " " << duration
        << std::endl;

    // Update to next step
    current_t += sim_dt;
    current_x = ddp_problem->simulate(current_x, current_u, sim_dt);
    current_u_list = ddp_solver->controlData().u_list;
  }

  // Final check
  const DDPProblem::InputDimVector & current_u = ddp_solver->controlData().u_list[0];
  DDPProblem::StateDimVector current_ref_x;
  DDPProblem::InputDimVector current_ref_u;
  ref_func(current_t, current_ref_x, current_ref_u);
  DDPProblem::StateDimVector x_tolerance;
  x_tolerance << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
  DDPProblem::InputDimVector u_tolerance;
  u_tolerance << 0.5, 0.5, 10.0, 10.0;
  for(int i = 0; i < current_x.size(); i++)
  {
    EXPECT_LT(std::abs(current_x[i] - current_ref_x[i]), x_tolerance[i])
        << "[TestMpcCartWalk] Violate final check for x[" << i << "]." << std::endl;
  }
  for(int i = 0; i < current_u.size(); i++)
  {
    EXPECT_LT(std::abs(current_u[i] - current_ref_u[i]), u_tolerance[i])
        << "[TestMpcCartWalk] Violate final check for u[" << i << "]." << std::endl;
  }

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path
            << "\" u 1:2 w lp, \"\" u 1:6 w lp, \"\" u 1:10 w lp, \"\" u 1:18 w l lw 2, \"\" u 1:22 w l lw 2 # pos_x\n"
            << "  plot \"" << file_path << "\" u 1:4 w lp, \"\" u 1:11 w lp, \"\" u 1:23 w l lw 2 # pos_y\n"
            << "  plot \"" << file_path << "\" u 1:8 w lp # obj_theta\n"
            << "  plot \"" << file_path << "\" u 1:12 w lp, \"\" u 1:13 w lp # obj_force\n"
            << "  plot \"" << file_path << "\" u 1:26 w lp # ddp_iter\n"
            << "  plot \"" << file_path << "\" u 1:27 w lp # computation_time\n";
}

int main(int argc, char ** argv)
{
  // Setup ROS
  ros::init(argc, argv, "test_mpc_cart_walk");

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
