#include "myelin_shim.h"

#include "spiking_network.cu"

extern "C" int myelin_launch_gif_step_weighted_f16(
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
    int n_inputs)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* membrane_ptr = reinterpret_cast<float*>(membrane);
    float* adaptation_ptr = reinterpret_cast<float*>(adaptation);
    const half* weights_ptr = reinterpret_cast<const half*>(weights);
    const float* input_spikes_ptr = reinterpret_cast<const float*>(input_spikes);
    unsigned int* refractory_ptr = reinterpret_cast<unsigned int*>(refractory);
    unsigned int* spikes_out_ptr = reinterpret_cast<unsigned int*>(spikes_out);

    gif_step_weighted_f16<<<grid_x, block_x, shared_bytes, cuda_stream>>>(
        membrane_ptr,
        adaptation_ptr,
        weights_ptr,
        input_spikes_ptr,
        refractory_ptr,
        spikes_out_ptr,
        n_neurons,
        n_inputs);

    return static_cast<int>(cudaGetLastError());
}

extern "C" int myelin_launch_saaq_find_best_walker(
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
    float adaptation_scale)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const float* membrane_ptr = reinterpret_cast<const float*>(membrane);
    const float* adaptation_ptr = reinterpret_cast<const float*>(adaptation);
    float* partial_scores_ptr = reinterpret_cast<float*>(partial_scores);
    unsigned int* partial_walkers_ptr = reinterpret_cast<unsigned int*>(partial_walkers);
    unsigned int* best_walker_ptr = reinterpret_cast<unsigned int*>(best_walker_out);

    saaq_find_best_walker<<<grid_x, block_x, shared_bytes, cuda_stream>>>(
        membrane_ptr,
        adaptation_ptr,
        partial_scores_ptr,
        partial_walkers_ptr,
        n_neurons,
        adaptation_scale);

    saaq_reduce_partials_f16<<<1u, 32u, 0u, cuda_stream>>>(
        partial_scores_ptr,
        partial_walkers_ptr,
        best_walker_ptr,
        static_cast<int>(grid_x));

    return static_cast<int>(cudaGetLastError());
}
