/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>
#include <torch/torch.h>

namespace DDMPC
{
/*! \brief Row major version of Eige::MatrixXf. */
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/*! \brief Row major version of Eige::MatrixXd. */
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*! \brief Convert to torch::Tensor.
 *  \param mat input (Eigen::MatrixXf)
 *
 *  Even if the matrix of colum major is passed as an argument, it is automatically converted to row major.
 */
inline torch::Tensor toTorchTensor(const MatrixXfRowMajor & mat)
{
  return torch::from_blob(const_cast<float *>(mat.data()), {mat.rows(), mat.cols()}).clone();
}

/*! \brief Convert to Eigen::MatrixXf.
 *  \param tensor input (torch::Tensor)
 */
inline Eigen::MatrixXf toEigenMatrix(const torch::Tensor & tensor)
{
  assert(tensor.dim() == 2);

  float * tensor_data_ptr = const_cast<float *>(tensor.data_ptr<float>());
  return Eigen::MatrixXf(Eigen::Map<MatrixXfRowMajor>(tensor_data_ptr, tensor.size(0), tensor.size(1)));
}
} // namespace DDMPC
