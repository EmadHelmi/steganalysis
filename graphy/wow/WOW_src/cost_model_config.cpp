#include "cost_model_config.h"
#include "base_cost_model_config.h"
#include <vector>
#include "wavelets.h"
#include "exception.hpp"
#include "mat2D.h"

cost_model_config::cost_model_config(float payload, bool verbose, float p, int waveletNumber, unsigned int stc_constr_height, int randSeed) : base_cost_model_config(payload, verbose, stc_constr_height, randSeed)
{
	this->p = p;
	set_filters(waveletNumber);	
}

cost_model_config::~cost_model_config()
{
	delete this->lpdf;
	delete this->Tlpdf;
	delete this->hpdf;
	delete this->Thpdf;
}

void cost_model_config::set_filters(int waveletNumber)
{
	waveletEnum wavelet;
	switch (waveletNumber) 
	{
		case 1: wavelet = daubechies8; break;
		default: throw exception("Unknown wavelet number.");
	}

	std::pair<mat2D<float> *, mat2D<float> *> filters = Wavelets::GetWavelets(wavelet);
	this->lpdf = filters.first;
	this->hpdf = filters.second;
	this->Tlpdf = mat2D<float>::Transpose(this->lpdf);
	this->Thpdf = mat2D<float>::Transpose(this->hpdf);

	if (this->lpdf->cols > this->hpdf->cols) this->padsize = this->lpdf->cols;
	else this->padsize = this->hpdf->cols;
}