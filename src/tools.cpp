#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse(4);
	rmse << 0,0,0,0;
	float n = estimations.size();
	VectorXd one_div_n(4);
	one_div_n << 1/n, 1/n, 1/n, 1/n;

	// check the validity of the inputs
  if (estimations.size() != ground_truth.size() || ground_truth.size() == 0) {
      cout << "CalculateRMSE: Error, bad input data" << endl;
  }

	// accumulate squared residuals
  VectorXd sq_diff(4);
	for(unsigned i=0; i < estimations.size(); ++i) {
        VectorXd diff = estimations[i] - ground_truth[i];
        sq_diff = diff.array() * diff.array();
        rmse += sq_diff;
	}
  // mean and squareroot
	rmse = one_div_n.array() * rmse.array();
	rmse = rmse.array().sqrt();

	return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	// Recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	// Check division by zero
	if(fabs(c1) < 0.0001) {
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	// Compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}

VectorXd Tools::CartesianToPolar(VectorXd & x_cartesian) {
	VectorXd x_polar(3);

	// Recover state parameters
	float px = x_cartesian(0);
	float py = x_cartesian(1);
	float vx = x_cartesian(2);
	float vy = x_cartesian(3);

	float px_2 = px * px;
	float py_2 = py * py;
	float psquare_sum = px_2 + py_2;
	float h1 = sqrt(psquare_sum);
	float h2  =atan2(py, px);
	float h3 = (px * vx + py * vy) / h1;

	x_polar << h1, h2, h3;
	return x_polar;
}
