/*
 * linedistance.h
 * Implement distance metrics in paper:
 * Trajectory Clustering: A Partition-and-Group Framework
 *
 *  Created on: Mar 25, 2018
 *      Author: me
 */

#ifndef LINEDISTANCE_H_
#define LINEDISTANCE_H_
#include <armadillo>
#include <algorithm>


double twodcross(arma::vec v1, arma::vec v2);

template<int Dimension=2>
class LineDistance
{
public:
	LineDistance() {}

	template<typename PointType, typename VecTypeB>
	static double point2line_distance(const PointType& a, const VecTypeB& b) {
		double linelength = arma::norm(b.head(Dimension) - b.tail(Dimension));
		return linelength == 0 ? 0.0 : twodcross(a - b.head(Dimension),
				b.head(Dimension) - b.tail(Dimension)) / linelength;
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
		VecTypeA shorter = a;
		VecTypeB longer = b;
		
		double alen = arma::norm(a.tail(Dimension) - a.head(Dimension));
		double blen = arma::norm(b.tail(Dimension) - b.head(Dimension));
		double shortlen = alen;
		double longlen = blen;
		if (alen > blen) {
			shorter = b;
			longer = a;
			std::swap(shortlen, longlen);
		}
		if (longlen != 0) {
			double l1 = std::min(
					std::abs(
							arma::dot(shorter.head(Dimension) - longer.head(Dimension),
									longer.tail(Dimension) - longer.head(Dimension))),
					std::abs(
							arma::dot(shorter.head(Dimension) - longer.tail(Dimension),
									longer.tail(Dimension) - longer.head(Dimension)))) / longlen;

			double l2 = std::min(
					std::abs(
							arma::dot(shorter.tail(Dimension) - longer.head(Dimension),
									longer.tail(Dimension) - longer.head(Dimension))),
					std::abs(
							arma::dot(shorter.tail(Dimension) - longer.tail(Dimension),
									longer.tail(Dimension) - longer.head(Dimension)))) / longlen;
			return std::min(l1, l2);
		} else {
			return 0;
		}
	}

	template<typename VecTypeA, typename VecTypeB>
	static double angular_distance(const VecTypeA& a, const VecTypeB& b) {
		double alen = arma::norm(a.head(Dimension) - a.tail(Dimension));
		double blen = arma::norm(b.head(Dimension) - b.tail(Dimension));

		double shorter_distance = std::min(alen, blen);
		double longer_distance = std::max(alen, blen);

		double distance = arma::dot(a.head(Dimension) - a.tail(Dimension),
				b.head(Dimension) - b.tail(Dimension));
		if (distance < 0) {
			return shorter_distance;
		} else {
			return longer_distance == 0 ? 0.0 : twodcross(
							a.head(Dimension) - a.tail(Dimension),
							b.head(Dimension) - b.tail(Dimension)) / longer_distance;
		}

	}

	template<typename VecTypeA, typename VecTypeB>
	static double Evaluate(const VecTypeA& a, const VecTypeB& b)
	{
		return angular_distance(a, b) + parallel_distance(a, b) + vertical_distance(a, b);
	}

};




#endif /* LINEDISTANCE_H_ */
