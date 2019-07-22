#include "base_cost_model.h"
#include "cost_model.h"
#include "mat2D.h"
#include "cost_model_config.h"
#include "base_cost_model_config.h"
#include <math.h>
#include <float.h>

cost_model::cost_model(mat2D<int>* cover, cost_model_config* config) : base_cost_model(cover, (base_cost_model_config *)config)
{
	this->config = config;
	float wetCost = pow(float(10), 10);

	// Create padded image
	mat2D<int> * cover_padded_int = mat2D<int>::Padding_Mirror(cover, config->padsize, config->padsize);
	mat2D<float>* cover_padded_double = mat2D<float>::Retype_int2float(cover_padded_int);
	delete cover_padded_int;

	// Compute residuals - wavelet sub-bands
	mat2D<float> * R_LH = mat2D<float>::Convolution_Same_basicFilters(cover_padded_double, config->Tlpdf, config->hpdf);
	mat2D<float> * R_HL = mat2D<float>::Convolution_Same_basicFilters(cover_padded_double, config->Thpdf, config->lpdf);
	mat2D<float> * R_HH = mat2D<float>::Convolution_Same_basicFilters(cover_padded_double, config->Thpdf, config->hpdf);
	delete cover_padded_double;

	mat2D<float> * R_LH_inv = mat2D<float>::InvertValues(mat2D<float>::AddValue(mat2D<float>::AbsoluteValue(R_LH), config->sigma));
	mat2D<float> * R_HL_inv = mat2D<float>::InvertValues(mat2D<float>::AddValue(mat2D<float>::AbsoluteValue(R_HL), config->sigma));
	mat2D<float> * R_HH_inv = mat2D<float>::InvertValues(mat2D<float>::AddValue(mat2D<float>::AbsoluteValue(R_HH), config->sigma));
    delete R_LH; delete R_HL; delete R_HH;

	mat2D<float> * xi_LH = mat2D<float>::Correlation_Same_basicFilters(R_LH_inv, mat2D<float>::AbsoluteValue(config->Tlpdf), mat2D<float>::AbsoluteValue(config->hpdf));
	mat2D<float> * xi_HL = mat2D<float>::Correlation_Same_basicFilters(R_HL_inv, mat2D<float>::AbsoluteValue(config->Thpdf), mat2D<float>::AbsoluteValue(config->lpdf));
	mat2D<float> * xi_HH = mat2D<float>::Correlation_Same_basicFilters(R_HH_inv, mat2D<float>::AbsoluteValue(config->Thpdf), mat2D<float>::AbsoluteValue(config->hpdf));
	delete R_LH_inv; delete R_HL_inv; delete R_HH_inv;

	mat2D<float> * xi_LH_no_padding = mat2D<float>::Submatrix(xi_LH, config->padsize+1, xi_LH->rows-config->padsize, config->padsize+1, xi_LH->cols-config->padsize);
	mat2D<float> * xi_HL_no_padding = mat2D<float>::Submatrix(xi_HL, config->padsize+1, xi_HL->rows-config->padsize, config->padsize+1, xi_HL->cols-config->padsize);
	mat2D<float> * xi_HH_no_padding = mat2D<float>::Submatrix(xi_HH, config->padsize+1, xi_HH->rows-config->padsize, config->padsize+1, xi_HH->cols-config->padsize);
	delete xi_LH; delete xi_HL; delete xi_HH;

	for (int r=0; r<cover->rows; r++)
	{
		for (int c=0; c<cover->cols; c++)
		{
			float rho = xi_LH_no_padding->Read(r, c) + xi_HL_no_padding->Read(r, c) + xi_HH_no_padding->Read(r, c);
			// pixel_costs[0] is the cost of -1, pixel_costs[0] is the cost of no change, pixel_costs[0] is the cost of +1
			float* pixel_costs = costs + ((c+r*cover->cols)*3);

			if (rho > wetCost) rho = wetCost;

			int coverVal = cover->Read(r, c);

			if (coverVal == 0) pixel_costs[0] = wetCost;
			else pixel_costs[0] = rho;

			pixel_costs[1] = 0;

			if (coverVal == 255) pixel_costs[2] = wetCost;
			else pixel_costs[2] = rho;
		}
	}

	delete xi_LH_no_padding; delete xi_HL_no_padding; delete xi_HH_no_padding;
}

cost_model::~cost_model()
{
}