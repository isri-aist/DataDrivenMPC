/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>

namespace DDMPC
{
/*! \brief Class of standardization (i.e., mean removal and variance scaling).
    \tparam Scalar scalar type
 */
template<class Scalar>
class StandardScaler
{
public:
  /** \brief Type of matrix. */
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  /** \brief Type of row vector. */
  using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

public:
  /*! \brief Constructor.
      \param data_all all data to calculate standardization coefficients
  */
  StandardScaler(const Matrix & data_all)
  {
    mean_vec_ = calcMean(data_all);
    stddev_vec_ = calcStddev(data_all, mean_vec_);
  }

  /*! \brief Apply standardization.
      \param data data to apply standardization
   */
  Matrix apply(const Matrix & data) const
  {
    return (data.rowwise() - mean_vec_).array().rowwise() / stddev_vec_.array();
  }

  /*! \brief Calculate mean. */
  static RowVector calcMean(const Matrix & data_all)
  {
    return data_all.colwise().mean();
  }

  /*! \brief Calculate standard deviation. */
  static RowVector calcStddev(const Matrix & data_all, const RowVector & mean)
  {
    return ((data_all.rowwise() - mean).cwiseAbs2().colwise().sum() / (data_all.rows() - 1)).cwiseSqrt();
  }

public:
  //! Mean of data
  RowVector mean_vec_;

  //! Standard deviation of data
  RowVector stddev_vec_;
};
} // namespace DDMPC
