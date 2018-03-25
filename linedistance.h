/*
 * linedistance.h
 *
 *  Created on: Mar 25, 2018
 *      Author: me
 */

#ifndef LINEDISTANCE_H_
#define LINEDISTANCE_H_
#include <armadillo>
#include <algorithm>

template<int Dimension=2>
class LineDistance
{
public:
	LineDistance() {}

	template<typename VecTypeA, typename VecTypeB>
	static double Evaluate(const VecTypeA& a, const VecTypeB& b)
	{
		// Return the L2 norm of the difference between the points, which is the
		// same as the L2 distance.
		return arma::norm(a - b);
	}

	template<typename PointType, typename VecTypeB>
	static double point2line_distance(const PointType& a, const VecTypeB& b) {
		double linelength = arma::norm(b.head(Dimension) - b.tail(Dimension));
		return linelength == 0 ? 0 : arma::norm(arma::cross(a - b.head(Dimension),
				b.head(Dimension) - b.tail(Dimension))) / linelength;

	}

	template<typename VecTypeA, typename VecTypeB>
	static double vertical_distance(const VecTypeA& a, const VecTypeB& b) {
		double lengtha = arma::norm(a.head(Dimension) - a.tail(Dimension));
		double lengthb = arma::norm(b.head(Dimension) - b.tail(Dimension));

		double distance = 0.0;
		double distance1 = 0.0;
		double distance2 = 0.0;

		if (lengtha > lengthb) {
			distance1 = point2line_distance(b.head(Dimension), a);
			distance2 = point2line_distance(b.tail(Dimension), a);
			distance = distance1 + distance2;
		} else {
			distance1 = point2line_distance(a.head(Dimension), b);
			distance2 = point2line_distance(a.tail(Dimension), b);
			distance = distance1 + distance2;
		}

		return distance == 0 ? 0 : (distance1 * distance1 + distance2 * distance2) / distance;
	}

	template<typename VecTypeA, typename VecTypeB>
	static double parallel_distance(const VecTypeA& a, const VecTypeB& b) {
		double l2square = arma.dot(b.tail(Dimension) - b.head(Dimension),
				a.tail(Dimension) - a.head(Dimension));
		if (l2square != 0) {
			double l1 = arma::norm(arma::dot(a.head(Dimension) - b.head(Dimension),
					b.tail(Dimension) - a.head(Dimension)) /
					l2square * (b.tail(Dimension) - b.head(Dimension)));
		}
	}

	template<typename VecTypeA, typename VecTypeB>
	static double angular_distance(const VecTypeA& a, const VecTypeB& b) {
		double alen = arma::norm(a.head(Dimension) - a.tail(Dimension));
		double blen = arma::norm(b.head(Dimension) - b.tail(Dimension));

		double shorter_distance = std::min(alen, blen);
		double longer_distance = std::max(alen, blen);

		double distance = arma::dot(a.head(Dimension) - a.tail(Dimension), b.head(Dimension) - b.tail(Dimension));
		if (distance < 0) {
			return shorter_distance;
		} else {
			return longer_distance == 0 ? 0.0 : arma::norm(
					arma::cross(
							a.head(Dimension) - a.tail(Dimension),
							b.head(Dimension) - b.tail(Dimension))) / longer_distance;
		}

	}

};




#endif /* LINEDISTANCE_H_ */
