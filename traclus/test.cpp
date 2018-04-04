/*
 * test.cpp
 *
 *  Created on: 2018年3月26日
 *      Author: me
 */
#include "traclus.h"
#include <mlpack/core/data/load.hpp>

int main(int argc, const char *argv[]) {
	arma::mat segments;
	if (argc > 1) {
		mlpack::data::Load(argv[1], segments);
		mlpack::data::Save("clusterid.csv", segment_clustering(segments, 0.01, 10));
	}
}

