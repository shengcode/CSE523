/*
 * test.cpp
 *
 *  Created on: 2018年3月26日
 *      Author: me
 */
#include "linedistance.h"
#include "traclus.h"
#include <mlpack/core/data/load.hpp>

int main(int argc, const char *argv[]) {
	arma::mat segments;
	arma::vec a={0.07650005,  0.86486883,  0.18464193,  0.17619998};
	arma::vec b={0.4078932 ,  0.95847937,  0.19174876,  0.20357427};
	arma::vec c={0.12693725,  0.47661061,  0.82950264,  0.8107738};

	double c3 = LineDistance<2>::Evaluate(a, b);
	double c1 = LineDistance<2>::Evaluate(a, c);
	double c2 = LineDistance<2>::Evaluate(c, b);

	if (argc > 1) {
		mlpack::data::Load(argv[1], segments);
		mlpack::data::Save("clusterid.csv", segment_clustering(segments, 0.01, 10));
	}
}

