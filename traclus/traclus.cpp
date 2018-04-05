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
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "linedistance.h"
#include "traclus.h"
#include <mlpack/core.hpp>
#include <iostream>
#include <armadillo>

arma::vec segment_clustering(const arma::mat& segments, double eps, int minlines) {
	arma::vec clusters = -arma::ones<arma::vec>(segments.n_cols);
	std::queue<arma::vec> q();
	std::cout << "Begin building tree" << std::endl;
	mlpack::neighbor::NeighborSearch<mlpack::neighbor::NearestNeighborSort, LineDistance<2>,
	arma::mat, mlpack::tree::BallTree > indexer(segments);

	std::cout << "Begin clustering" << std::endl;

	int clusterid = 0;
	std::queue<size_t> tobeclassified;
	for (size_t i = 0; i < segments.n_cols; ++i) {
		if (clusters(i) == -1) {
			arma::mat segment = segments.col(i);
			arma::Mat<size_t> neighbors;
			arma::mat distances;
			indexer.Search(segment, minlines, neighbors, distances);
			if (distances.max() <= eps) {
				clusters(i) = clusterid;
				for (size_t j = 0; j < neighbors.n_cols; ++j) {
					clusters(neighbors(j)) = clusterid;
					tobeclassified.push(neighbors(j));
				}
				while (!tobeclassified.empty()) {
					auto k = tobeclassified.front();
					tobeclassified.pop();
					arma::Mat<size_t> kneighbors;
					arma::mat kdistances;
					indexer.Search(segments.col(k), minlines, kneighbors, kdistances);
					if (kdistances.max() <= eps) {
						for (size_t idx = 0; idx < kneighbors.n_cols; ++idx) {
							if (clusters(kneighbors(idx)) < 0) {
								if (clusters(kneighbors(idx)) == -1) {
									tobeclassified.push(kneighbors(idx));
								}
								clusters(kneighbors(idx)) = clusterid;
							}
						}
					}
				}
				clusterid++;
			} else {
				clusters(i) = -2;
			}
		}
	}
	return clusters;
}




