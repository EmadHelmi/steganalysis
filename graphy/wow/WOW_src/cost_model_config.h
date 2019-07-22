#ifndef CONFIG_H_
#define CONFIG_H_

#include "base_cost_model_config.h"
#include <vector>
#include "mat2D.h"

class cost_model_config : public base_cost_model_config
{
public:
	float p;
	mat2D<float> * lpdf;
	mat2D<float> * hpdf;
	mat2D<float> * Tlpdf;
	mat2D<float> * Thpdf;
	int padsize;

	cost_model_config(float payload, bool verbose, float p, int wavelet, unsigned int stc_constr_height, int randSeed);
	~cost_model_config();

private:
	void set_filters(int wavelet);
};
#endif