#include "ggml/ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

struct xor_model {
    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;
    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;
    struct ggml_context * ctx;
};

bool xor_model_load(const std::string & fname, xor_model & model) {
    struct gguf_init_params params = {
            /*.no_alloc   =*/ false,
            /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.fc1_weight = ggml_get_tensor(model.ctx, "fc1.weight");
    model.fc1_bias = ggml_get_tensor(model.ctx, "fc1.bias");
    model.fc2_weight = ggml_get_tensor(model.ctx, "fc2.weight");
    model.fc2_bias = ggml_get_tensor(model.ctx, "fc2.bias");
    return true;
}

int xor_eval(
        const xor_model & model,
        std::vector<float> input_data
)
{
    static size_t buf_size = 1000 * input_data.size() * sizeof(float) * 32;
    static void * buf = malloc(buf_size);
    struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ buf,
            /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2);
    memcpy(input->data, input_data.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input), model.fc1_bias);
    fc1 = ggml_hardsigmoid(ctx0, fc1);

    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, fc1), model.fc2_bias);
    fc2 = ggml_hardsigmoid(ctx0, fc2);

    ggml_build_forward_expand(gf, fc2);
    ggml_graph_compute_with_ctx(ctx0, gf, 1);

    const float * fc2_data = ggml_get_data_f32(fc2);
    int prediction = (fc2_data[0] > 0.5) ? 1 : 0;

    return prediction;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s models/xornet/xornet.gguf\n", argv[0]);
        exit(0);
    }

    xor_model model;

    // load the model
    if (!xor_model_load(argv[1], model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
        return 1;
    }

    fprintf(stdout, "%s: predicted output is %d for input (0,0)\n", __func__, xor_eval(model, {0.0f, 0.0f}));
    fprintf(stdout, "%s: predicted output is %d for input (0,1)\n", __func__, xor_eval(model, {0.0f, 1.0f}));
    fprintf(stdout, "%s: predicted output is %d for input (1,0)\n", __func__, xor_eval(model, {1.0f, 0.0f}));
    fprintf(stdout, "%s: predicted output is %d for input (1,1)\n", __func__, xor_eval(model, {1.0f, 1.0f}));


    return 0;
}