#ifndef INFO_THEORY_H_
#define INFO_THEORY_H_

#include <cmath>
#include <limits>
#include <valarray>
// #include "acml_mv.h"

typedef std::valarray<float> va_float;
typedef unsigned int uint;

inline double bin_entropy(double x);
inline float bin_entropyf(float x);
float inv_bin_entropyf(float y);
double inv_bin_entropy(double y);

inline double qary_entropy(double x, double q);
inline float  qary_entropyf(float x, float q);
float inv_qary_entropyf(float y, float q);
double inv_qary_entropy(double y, double q);

// binary entropy function in double precission
inline double bin_entropy(double x) {

    double const LOG2 = log(2.0);
    double const EPS = std::numeric_limits<double>::epsilon();
    double z;

    if ((x<EPS) || ((1-x)<EPS)) {
        return 0;
    } else {
        z = (-x*log(x)-(1-x)*log(1-x))/LOG2;
        return z;
    }
}

// binary entropy function in single precision
inline float bin_entropyf(float x) {

    float const LOG2 = log(2.0f);
    float const EPS = std::numeric_limits<float>::epsilon();
    float z;

    if ((x<EPS) || ((1-x)<EPS)) {
        return 0;
    } else {
        z = (-x*log(x)-(1-x)*log(1-x))/LOG2;
        return z;
    }
}

float entropy(const va_float &prob_distribution);

/*
inline float entropy_gibbs_distribution(const va_float &weights, float beta) {

	float const LOG2 = log(2.0);
	float const EPS = std::numeric_limits<float>::epsilon();
	float z = 0;
	va_float flip_prob(weights);
	double sum = 0;

	for(uint i=0;i<flip_prob.size();i++) {
		flip_prob[i] = fastexpf(-beta*weights[i]); sum += flip_prob[i];
	}
	flip_prob /= sum; // normalize to a probability distribution

	for(uint i=0;i<flip_prob.size();i++) {
		if (flip_prob[i]>EPS)
			z += beta*flip_prob[i]*weights[i];
	}
	return (z+fastlogf(sum))/LOG2;
}
*/

#endif
