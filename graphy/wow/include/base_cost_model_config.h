#ifndef BASE_COST_MODEL_CONFIG_H_
#define BASE_COST_MODEL_CONFIG_H_

class base_cost_model_config
{
public:
	float payload;
	bool verbose;
	unsigned int stc_constr_height;
	int randSeed;

	base_cost_model_config(float payload, bool verbose, unsigned int stc_constr_height, int randSeed);
	~base_cost_model_config();
};
#endif