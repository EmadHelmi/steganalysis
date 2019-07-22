#include <time.h>
#include <memory.h>
#include <limits>
#include <algorithm>
#include <valarray>
#include <iostream>
#include <string>
#include <math.h>

#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "mi_embedder.h"
#include "info_theory.h"
#include "stc_ml_c.h"

#include "base_cost_model.h"
#include "mat2D.h"
#include "exception.hpp"

typedef unsigned int uint;
typedef std::valarray< float > va_float;

/* ********* MUTUALLY INDEPENDENT EMBEDDING ALGORITHM ****************************************************************************** */
int mi_emb_sample_from_distribution( float* P, double random_number ) {

	if (random_number < P[0]) return -1;
	else if (random_number < P[0]+P[2]) return 1;
	else return 0;
}

float* cost2probability(float* changeCosts, float lambda) 
{
	float* P = new float[3];

	float minCost = changeCosts[0];
	for (int k=1; k<3; k++) if (changeCosts[k] < minCost) minCost = changeCosts[k];

	float div = exp(-lambda*(changeCosts[0]-minCost)) + exp(-lambda*(changeCosts[1]-minCost)) + exp(-lambda*(changeCosts[2]-minCost));
	P[0] = exp(-lambda*(changeCosts[0]-minCost)) / div;
	P[1] = exp(-lambda*(changeCosts[1]-minCost)) / div;
	P[2] = exp(-lambda*(changeCosts[2]-minCost)) / div;

	return P;
}

/* CALCULATE AVERAGE PAYLOAD FOR GIVEN LAMBDA
 */
float mi_emb_calc_average_payload( base_cost_model* m, float lambda ) 
{
	float const LOG2 = log( 2.0f );

	float H = 0;
	for (int i=0; i<m->rows*m->cols; i++)
	{
		float* changeCosts = m->costs + (i*3);
		float* P = cost2probability(changeCosts, lambda) ;

		if (P[0] > DBL_MIN) H -= P[0]*log(P[0]);
		if (P[1] > DBL_MIN) H -= P[1]*log(P[1]);
		if (P[2] > DBL_MIN) H -= P[2]*log(P[2]);

		delete P;   
	}
	H /= LOG2;
	return H;
}

mat2D<int>* mi_emb_simulate_pls_embedding(base_cost_model* m, float alpha, uint seed, float &lambda, float &distortion, float &alpha_out ) {

    if ( lambda < 0 ) lambda = mi_emb_calculate_lambda_from_payload( m, alpha, 1, alpha_out );
	mat2D<int>* stego = new mat2D<int>(m->rows, m->cols);
    boost::mt19937 generator( seed );
    boost::variate_generator< boost::mt19937&, boost::uniform_real< > > rng( generator, boost::uniform_real< >( 0, 1 ) );

    for ( int i = 0; i < m->rows; i++ ) {
        for ( int j = 0; j < m->cols; j++ ) {
			float* changeCosts = m->costs + (3*(i*m->cols + j));
			
			float* P = cost2probability(changeCosts, lambda);

            // sample from distribution flip_prob
            int stego_noise = mi_emb_sample_from_distribution(P, rng());
			delete P;
			stego->Write(i, j, m->cover->Read(i,j) + stego_noise);

            // update distortion
            distortion += changeCosts[stego_noise + 1];
        }
    }

    return stego;
}

float mi_emb_calculate_lambda_from_payload(base_cost_model* m, float rel_payload, float lambda_init, float &alpha_out ) 
{
    float alpha1, alpha2, alpha3, lambda1, lambda2, lambda3;
    int j = 0;
    uint iterations = 0;

    lambda1 = 0;
    alpha1 = 1;
    if ( lambda_init < 0 ) 
	{
        lambda3 = 1e+3;
        alpha3 = 10; // this is just an initial value
    } 
	else 
	{
        lambda3 = 0.5f * lambda_init;
        alpha3 = 10; // this is just an initial value
    }
    lambda2 = lambda_init;
    while ( alpha3 > rel_payload ) {
        lambda3 *= 2;
        alpha3 = mi_emb_calc_average_payload( m, lambda3 ) / (m->rows * m->cols);
        j++;
        // lambda is probably unbounded => it seems that we cannot find lambda such that
        // relative payload will be smaller than requested. Binary search does not make sence here.
        if ( j > 100 ) return lambda3; // throw exception("Number of trials in lambda binary search exceeded.");
        iterations++;
    }
    alpha2 = alpha3;
    while ( alpha1 - alpha3 > rel_payload * 1e-2 ) { // iterative search for parameter lambda
        lambda2 = lambda1 + (lambda3 - lambda1) / 2;
        alpha2 = 0;
        alpha2 = mi_emb_calc_average_payload( m, lambda2 ) / (m->rows * m->cols);
        if ( alpha2 < rel_payload ) {
            lambda3 = lambda2;
            alpha3 = alpha2;
        } else {
            lambda1 = lambda2;
            alpha1 = alpha2;
        }
        iterations++; // this is for monitoring the number of iterations
		if ( iterations > 100 )
		{
			alpha_out = alpha2;
			return lambda2;
		}
    }

    alpha_out = alpha2;
    return lambda2;
}

mat2D<int>* mi_emb_stc_pls_embedding(base_cost_model* m, float alpha, uint seed, uint stc_constr_height, uint stc_max_trails, float &distortion,
        float &alpha_out, float &coding_loss, uint &stc_trials_used) 

{
    distortion = 0;
    uint n = m->rows * m->cols;
    unsigned char *stego_pixels = new unsigned char[n];

    int *cover_px = new int[n];
    int *stego_px = new int[n];

    for ( int i = 0; i < m->rows; i++ ) 
	{
        for ( int j = 0; j < m->cols; j++ ) 
		{
			cover_px[i * m->cols + j] = m->cover->Read(i, j);
        }
    }

    boost::mt19937 generator( seed );
    boost::variate_generator< boost::mt19937&, boost::uniform_int< > > rng( generator, boost::uniform_int< >( 0, RAND_MAX ) );

    uint message_length = (uint) floor( alpha * n );
    unsigned char *message = new unsigned char[message_length];
    for ( uint i = 0; i < message_length; i++ ) // generate random message
        message[i] = rng() % 2;
    stc_trials_used = stc_max_trails;
    uint *num_msg_bits = new uint[2]; // this array contains number of bits embedded into first and second layer

    try 
	{
        distortion = stc_pm1_pls_embed( n, cover_px, m->costs, message_length, message, stc_constr_height, F_INF, stego_px, num_msg_bits,
                stc_trials_used, &coding_loss );
    } 
	catch (...) 
	{
        num_msg_bits[0] = 0;
        num_msg_bits[1] = 0;
        distortion = 0;
        coding_loss = 0;
    }

    // check that the embedded message can be extracted
    // extract message from 'stego_array' into 'extracted_message' and use STCs with constr. height h
    unsigned char *extracted_message = new unsigned char[message_length];
    stc_ml_extract( n, stego_px, 2, num_msg_bits, stc_constr_height, extracted_message );

    // check the extracted message
    bool msg_ok = true;
    for ( uint k = 0; k < num_msg_bits[0] + num_msg_bits[1]; k++ ) {
        msg_ok &= (extracted_message[k] == message[k]);
        if ( !msg_ok ) throw exception( "ML_STC_ERROR: Extracted message differs in bit " );
    }

    if ( num_msg_bits[0] + num_msg_bits[1] > 0 ) {
        for ( uint i = 0; i < n; i++ )
            stego_pixels[i] = stego_px[i];
    } else {
        for ( uint i = 0; i < n; i++ )
            stego_pixels[i] = cover_px[i];
    }

	mat2D<int>* stego = new mat2D<int>(m->rows, m->cols);
    for ( int i = 0; i < m->rows; i++ ) 
	{
        for ( int j = 0; j < m->cols; j++ ) 
		{
			stego->Write(i, j, stego_pixels[i*m->cols+j]);
        }
    }
    alpha_out = (float) (num_msg_bits[0] + num_msg_bits[1]) / (float) n;

    delete[] stego_pixels;
    delete[] cover_px;
    delete[] stego_px;
    delete[] message;
    delete[] extracted_message;
    delete[] num_msg_bits;

    return stego;
}
