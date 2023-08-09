#ifndef AMREX_BACKGROUND_STREAM_H_
#define AMREX_BACKGROUND_STREAM_H_

#include <AMReX_BackgroundThread.H>
#include <AMReX_Config.H>
#include <AMReX_Gpu.H>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <future>

#define CU_CHECK(stmt)                                                            \
    do {                                                                          \
        CUresult result = (stmt);                                                 \
        if (CUDA_SUCCESS != result) {                                             \
            const char *str;                                                      \
            CUresult str_result = cuGetErrorString(result, &str);                 \
            if (str_result != CUDA_SUCCESS)                                       \
                fprintf(stderr, "[%s:%d] cu failed with unknown error %d\n",      \
                        __FILE__, __LINE__, result);                              \
            else                                                                  \
                fprintf(stderr, "[%s:%d] cu failed with %s \n",                   \
                        __FILE__, __LINE__, str);                                 \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

namespace amrex {

class BackgroundStream
    :
    public BackgroundThread
{
public:
    BackgroundStream ();
    ~BackgroundStream ();
    BackgroundStream (BackgroundStream const&) = delete;
    BackgroundStream (BackgroundStream &&) = delete;
    BackgroundStream& operator= (BackgroundStream const&) = delete;
    BackgroundStream& operator= (BackgroundStream &&) = delete;

    void cpuSubmit (std::function<void()>&& f);
    void gpuSubmit (std::function<void()>&& f);

    cudaStream_t& get_stream() { return gpu_stream; }

    void cpuSync ();
    void gpuSync ();
    void sync    ();

    // Want to make sure all jobs finish on CPU and GPU. 
    // Need a new one?
//    void Finish (); // Not required. Call this if you want all jobs to finish.

private:

    cudaStream_t gpu_stream;

    enum func_type { NONE, CPU, GPU };
    func_type previous = NONE;

    int op_value = 0;
    CUdeviceptr dptr;
    volatile int* hptr;

    std::queue<cudaEvent_t> events;

    std::mutex e_mtx;
};

}

#endif
