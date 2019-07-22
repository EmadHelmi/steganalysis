#include <vector>
#include "cost_model.h"
#include "mat2D.h"
#include "mi_embedder.h"
#include "cost_model_config.h"
#include <cstring>
#include <mex.h>

/*
	prhs[0] - uint8						- uint8 (unsigned int) matrix			- matrix with the image
	prhs[1] - payload					- single (float)
	prhs[2] - struct config
				config.p				- single (float)		- default -1	- holder norm parameter
				config.STC_h			- uint8 (insigned int)	- default 0		- 0 for optimal simulator, otherwise STC submatrix height (try 7-12)
				config.seed				- int					- default 0		- random seed
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	const mxArray *mat_cover;

	// Default config
	float c_payload;
	float c_p = -1;
	int c_randSeed = 0;
	unsigned int c_stc_constr_height = 0;

	if ((nrhs != 2) && (nrhs != 3))
		mexErrMsgTxt ("Two or three inputs are required.\n2 inputs - [image matrix (uint8)] [payload (single)] \n3 inputs - [image matrix (uint8)] [payload (single)] [struct config]");

	if  (mxIsClass(prhs[0], "uint8"))
		mat_cover = prhs[0];
	else
		mexErrMsgTxt ("The first input (cover image) must be a 'uint8' matrix.");

	if  ((mxIsClass(prhs[1], "single")) && (mxGetM(prhs[1])== 1) && (mxGetN(prhs[1])== 1))
		c_payload = (float)mxGetScalar(prhs[1]);
	else
		mexErrMsgTxt ("The second input (payload) must be a number of type 'single'.");
	if (nrhs == 3)
	{
		const mxArray *mat_config = prhs[2];

		int nfields = mxGetNumberOfFields(mat_config);
		if (nfields==0) mexErrMsgTxt ("The config structure is empty.");
		for(int fieldIndex=0; fieldIndex<nfields; fieldIndex++)
		{
			const char *fieldName = mxGetFieldNameByNumber(mat_config, fieldIndex);
			const mxArray *fieldContent = mxGetFieldByNumber(mat_config, 0, fieldIndex);

			if ((mxGetM(fieldContent)!= 1) || (mxGetN(fieldContent)!= 1))
				mexErrMsgTxt ("All config fields must be scalars.");
			// if every field is scalar
			if (strcmp(fieldName, "p") == 0)
				if (mxIsClass(fieldContent, "single")) c_p = (float)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.p' must be of type 'single'");
			if (strcmp(fieldName, "STC_h") == 0)
				if (mxIsClass(fieldContent, "uint32")) c_stc_constr_height = (unsigned int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.STC_h' must be of type 'uint32'");
			if (strcmp(fieldName, "seed") == 0)
				if (mxIsClass(fieldContent, "int32")) c_randSeed = (int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.seed' must be of type 'int32'");
		}
	}
	// create C cover matrix
	int rows = (int)mxGetM(mat_cover);
	int cols = (int)mxGetN(mat_cover);
	mat2D<int> *c_cover = new mat2D<int>(rows, cols);
	unsigned char *cover_array = (unsigned char *)mxGetData(mat_cover);

	for (int c=0; c<cols; c++)
	{
		for (int r=0; r<rows; r++)
		{
			c_cover->Write(r, c, (int)cover_array[r+c*rows]);
		}
	}
	// Embedding
	cost_model_config *c_config = new cost_model_config(c_payload, false, c_p, 1, c_stc_constr_height, c_randSeed);
	base_cost_model * model = (base_cost_model *)new cost_model(c_cover, c_config);
	float c_alpha_out, c_coding_loss_out = 0, c_distortion = 0;
	unsigned int c_stc_trials_used = 0;
	mat2D<int> * c_stego = model->Embed(c_alpha_out, c_coding_loss_out, c_stc_trials_used, c_distortion);

	delete model;
	delete c_config;
	// Create matlab stego matrix
	mwSize stegoSize[2];
	stegoSize[0] = (size_t)rows;
	stegoSize[1] = (size_t)cols;
	mxArray *mat_stego = mxCreateNumericArray(2, stegoSize, mxUINT8_CLASS, mxREAL);
	for (int c=0; c<cols; c++)
	{
		for (int r=0; r<rows; r++)
		{
			((unsigned char *)mxGetData(mat_stego))[r+c*rows] = (unsigned char)c_stego->Read(r, c);
		}
	}	
	
	mwSize scalarSize[2];
	scalarSize[0] = (size_t)1;
	scalarSize[1] = (size_t)1;
	plhs[0] = mat_stego;

	// Create distortion for matlab
	mxArray *mat_distortion = mxCreateNumericArray(2, scalarSize, mxSINGLE_CLASS, mxREAL);
	((float *)mxGetPr(mat_distortion))[0] = c_distortion;
	plhs[1] = mat_distortion;
} 