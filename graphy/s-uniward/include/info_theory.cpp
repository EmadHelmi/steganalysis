#include "info_theory.h"

// q-ary entropy function - see Section 8.6.2 in "Steganography in digital media..."
// double precision
inline double qary_entropy(double x, double q) {
    return bin_entropy(x) + x*log(q-1)/log(2.0);
}

// single precision
inline float qary_entropyf(float x, float q) {
    return bin_entropyf(x) + x*log(q-1)/log(2.0f);
}

template<class T> T binary_search(T x_min, T x_max, T y, T parameter, T (*f)(T,T), T precision) {

    T x1, x2, x3, y1, y2, y3;

    x1 = x_min; y1 = f(x1, parameter);
    x3 = x_max; y3 = f(x3, parameter);
    x2 = 0; // this is just an initialization
    while (y3-y1>precision) { // binary search
        x2 = x1+(x3-x1)/(T)2.0; y2 = f(x2, parameter);
        if (y>y2) {
            x1 = x2; y1 = y2;
        } else {
            x3 = x2; y3 = y2;
        }
    }

    return x2;
}

// inverse of entropy functions on interval [0,max]
float inv_bin_entropyf(float y)  { return binary_search<float>(0.0f, 0.5f, y, 2.0f, qary_entropyf, 1e-5f); }
double inv_bin_entropy(double y) { return binary_search<double>(0.0, 0.5, y, 2.0, qary_entropy, 1e-5); }
float inv_qary_entropyf(float y, float q)  { return binary_search<float>(0.0f, (q-1)/q, y, q, qary_entropyf, 1e-5f); }
double inv_qary_entropy(double y, double q){ return binary_search<double>(0.0f, (q-1)/q, y, q, qary_entropy, 1e-5f); }

float entropy(const va_float &prob_distribution) {

	float const LOG2 = log(2.0f);
	float const EPS = std::numeric_limits<float>::epsilon();
	float z = 0;

	for(uint i=0;i<prob_distribution.size();i++) {
		if (prob_distribution[i]>EPS)
			z += -prob_distribution[i]*log(prob_distribution[i])/LOG2;
	}
	return z;
}
