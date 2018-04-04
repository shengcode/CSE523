/*
 * traclus.h
 *
 *  Created on: 2018年3月26日
 *      Author: me
 */

#ifndef TRACLUS_H_
#define TRACLUS_H_
#include <armadillo>

arma::vec segment_clustering(const arma::mat& segments, double eps, int minlines);


#endif /* TRACLUS_H_ */
