/*
 * traclus.cpp
 *
 *  Created on: Mar 25, 2018
 *      Author: me
 */
#include <queue>
#include <mlpack/core.hpp>
#include <mlpack/core/tree/statistic.hpp>
#include <mlpack/core/tree/binary_space_tree/typedef.hpp>
#include <armadillo>
#include "linedistance.h"

arma::vec segment_clustering(const arma::mat& segments, double eps, double minlines) {
	auto clusterid = -arma::ones(segments.n_rows);
	std::queue<arma::vec> q();
	auto tree = mlpack::tree::BallTree<LineDistance<2>, mlpack::tree::EmptyStatistic, arma::mat>(segments);

	return clusterid;
}




