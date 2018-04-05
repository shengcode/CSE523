/*
 * linedistance.cpp
 *
 *  Created on: 2018年4月3日
 *      Author: me
 */
#include "linedistance.h"

double twodcross(arma::vec v1, arma::vec v2) {
	return std::abs(v1(0)*v2(1) - v1(1)*v2(0));
}


