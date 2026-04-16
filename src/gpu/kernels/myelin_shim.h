#ifndef MYELIN_SHIM_H
#define MYELIN_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

int myelin_launch_gif_step_weighted_f16(
    void* stream,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    void* membrane,
    void* adaptation,
    void* weights,
    void* input_spikes,
    void* refractory,
    void* spikes_out,
    int n_neurons,
    int n_inputs);

int myelin_launch_saaq_find_best_walker(
    void* stream,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    void* membrane,
    void* adaptation,
    void* partial_scores,
    void* partial_walkers,
    void* best_walker_out,
    int n_neurons,
    float adaptation_scale);

#ifdef __cplusplus
}
#endif

#endif  // MYELIN_SHIM_H
