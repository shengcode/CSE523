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
namespace {
template<typename Neighbor, typename MatType>
void search_eps(Neighbor& indexer, const MatType& segment, double eps, size_t minlines,
		arma::Mat< size_t > &neighbours, arma::mat &distances) {
	size_t numlines = minlines;
	size_t lower_num = minlines;
	size_t upper_num = 0;
	double diameter = -1;
	double diameter2 = -1;
	do {
		indexer.Search(segment, numlines, neighbours, distances);
		diameter = distances.max();
		if (diameter > eps) {
			upper_num = numlines;
			if (lower_num > 0) {
				numlines = (lower_num + upper_num) / 2;
			} else {
				numlines /= 2;
			}
		} else {
			lower_num = numlines;
			if (upper_num > 0) {
				numlines = (lower_num + upper_num) / 2;
			} else {
				numlines *= 2;
			}
		}
		if (numlines < minlines || abs(upper_num - lower_num) <= 1) {
			break;
		}
		diameter2 = diameter;
	} while(true);
	if (diameter > eps) {
		indexer.Search(segment, lower_num, neighbours, distances);
	}
}
}

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
			//indexer.Search(segment, minlines, neighbors, distances);
			search_eps(indexer, segment, eps, minlines, neighbors, distances);
			if (neighbors.size() >= minlines) {
				clusters(i) = clusterid;
				for (size_t j = 0; j < neighbors.n_rows; ++j) {
					clusters(neighbors(j)) = clusterid;
					tobeclassified.push(neighbors(j));
				}
				while (!tobeclassified.empty()) {
					auto k = tobeclassified.front();
					tobeclassified.pop();
					arma::Mat<size_t> kneighbors;
					arma::mat kdistances;
					search_eps(indexer, segments.col(k), eps, minlines, kneighbors, kdistances);
					if (kneighbors.size() >= minlines) {
						for (size_t idx = 0; idx < kneighbors.n_rows; ++idx) {
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




