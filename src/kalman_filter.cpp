#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // update the state by using Kalman Filter equations

  VectorXd y = z - (H_ * x_);
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  x_ = x_ + K * y;
  P_ = (I - (K * H_)) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;
  // update the state by using Extended Kalman Filter equations
  VectorXd h_x = tools.CartesianToPolar(x_);
  H_ = tools.CalculateJacobian(x_);
  VectorXd y = z - h_x;

  // keep phi between -pi and pi
  while (y(1) < -M_PI) {
    y(1) = y(1) + 2 * M_PI;
  };
  while (y(1) > M_PI) {
    y(1) = y(1) - 2 * M_PI;
  };

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  x_ = x_ + K * y;
  P_ = (I - (K * H_)) * P_;
}
