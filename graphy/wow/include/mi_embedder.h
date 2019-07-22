#ifndef MI_EMBEDDER_H_
#define MI_EMBEDDER_H_

#include <cfloat>
#include "exception.hpp"
#include "base_cost_model.h"
#include "mat2D.h"

typedef unsigned int uint;

/* ********* MUTUALLY INDEPENDENT EMBEDDING ALGORITHM ****************************************************************************** */

mat2D<int>* mi_emb_simulate_pls_embedding(base_cost_model *m, float alpha, uint seed, float &lambda, float &distortion, float &alpha_out );
mat2D<int>* mi_emb_stc_pls_embedding(base_cost_model *m, float alpha, uint seed, uint stc_constr_height, uint stc_max_trails, float &distortion, float &alpha_out, float &coding_loss_out, uint &stc_trials_used);
float mi_emb_calculate_lambda_from_payload(base_cost_model *m, float rel_payload, float lambda_init, float &alpha_out);
float mi_emb_calc_average_payload(base_cost_model *m, float lambda);

#endif // MI_EMBEDDER_H_
