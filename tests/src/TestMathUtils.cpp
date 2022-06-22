/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <DDMPC/MathUtils.h>

template<int DataDim>
void testStandardScaler()
{
  // Generate data
  Eigen::RowVector3d scale(10.0, 20.0, 30.0);
  Eigen::RowVector3d offset(0.0, 100.0, -200.0);
  Eigen::MatrixX3d data_all = (Eigen::MatrixX3d::Random(10000, 3) * scale.asDiagonal()).rowwise() + offset;
  Eigen::MatrixX3d train_data = data_all.topRows(static_cast<int>(0.7 * data_all.rows()));
  Eigen::MatrixX3d test_data = data_all.bottomRows(static_cast<int>(0.3 * data_all.rows()));

  // Apply standardization
  DDMPC::StandardScaler<double, DataDim> standard_scaler(train_data);
  Eigen::MatrixX3d standardized_train_data = standard_scaler.apply(train_data);
  Eigen::MatrixX3d standardized_test_data = standard_scaler.apply(test_data);

  // Check standardized train data
  {
    Eigen::RowVector3d mean = DDMPC::StandardScaler<double, DataDim>::calcMean(standardized_train_data);
    Eigen::RowVector3d stddev = DDMPC::StandardScaler<double, DataDim>::calcStddev(standardized_train_data, mean);
    EXPECT_LT(mean.norm(), 1e-10);
    EXPECT_LT((stddev.array() - 1.0).matrix().norm(), 1e-10);
  }

  // Check standardized test data
  {
    Eigen::RowVector3d mean = DDMPC::StandardScaler<double, DataDim>::calcMean(standardized_test_data);
    Eigen::RowVector3d stddev = DDMPC::StandardScaler<double, DataDim>::calcStddev(standardized_test_data, mean);
    EXPECT_LT(mean.norm(), 0.1);
    EXPECT_LT((stddev.array() - 1.0).matrix().norm(), 0.1);
  }

  // Check inverse standardization
  {
    Eigen::MatrixX3d restored_test_data = standard_scaler.applyInv(standardized_test_data);
    EXPECT_LT((test_data - restored_test_data).norm(), 1e-10);

    Eigen::Vector3d single_test_data = test_data.row(0).transpose();
    EXPECT_LT((single_test_data - standard_scaler.applyOneInv(standard_scaler.applyOne(single_test_data))).norm(),
              1e-10);
    EXPECT_LT((single_test_data - standard_scaler.applyOne(standard_scaler.applyOneInv(single_test_data))).norm(),
              1e-10);
  }
}

TEST(TestMathUtils, StandardScalerFixedSize)
{
  testStandardScaler<3>();
}

TEST(TestMathUtils, StandardScalerDynamicSize)
{
  testStandardScaler<Eigen::Dynamic>();
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
