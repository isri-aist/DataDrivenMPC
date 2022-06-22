/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>

namespace DDMPC
{
/*! \brief Class of standardization (i.e., mean removal and variance scaling).
    \tparam Scalar scalar type
    \tparam DataDim data dimension
 */
template<class Scalar, int DataDim>
class StandardScaler
{
public:
  /** \brief Type of matrix. */
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, DataDim>;

  /** \brief Type of column vector. */
  using Vector = Eigen::Matrix<Scalar, DataDim, 1>;

  /** \brief Type of row vector. */
  using RowVector = Eigen::Matrix<Scalar, 1, DataDim>;

public:
  /*! \brief Constructor.
      \param data_all all data to calculate standardization coefficients
  */
  StandardScaler(const Matrix & data_all)
  {
    mean_vec_ = calcMean(data_all);
    stddev_vec_ = calcStddev(data_all, mean_vec_).cwiseMax(1e-6); // Set minimum to avoid zero devision
  }

  /*! \brief Apply standardization.
      \param data data to apply standardization
   */
  Matrix apply(const Matrix & data) const
  {
    return (data.rowwise() - mean_vec_).array().rowwise() / stddev_vec_.array();
  }

  /*! \brief Apply standardization.
      \param data single data to apply standardization
   */
  Vector applyOne(const Vector & data) const
  {
    return (data - mean_vec_.transpose()).array() / stddev_vec_.transpose().array();
  }

  /*! \brief Apply inverse standardization.
      \param data data to apply inverse standardization
   */
  Matrix applyInv(const Matrix & data) const
  {
    return (data.array().rowwise() * stddev_vec_.array()).matrix().rowwise() + mean_vec_;
  }

  /*! \brief Apply inverse standardization.
      \param data single data to apply inverse standardization
   */
  Vector applyOneInv(const Vector & data) const
  {
    return data.cwiseProduct(stddev_vec_.transpose()) + mean_vec_.transpose();
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
