
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <thread>
#include <future>

#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif

struct gpu_promise {
    double sleep;
    double tz;
    std::promise<void> timestep;
};

#ifdef AMREX_USE_CUDA
// Mocking GPU callback function work
void CUDART_CB do_step_gpu(void* p)
{
    auto p_prom = reinterpret_cast<gpu_promise*>(p);

    amrex::Print() << "Beginning GPU work at " << amrex::second() - p_prom->tz << std::endl;
    amrex::Sleep(p_prom->sleep);
    amrex::Print() << "Ending GPU work at " << amrex::second() - p_prom->tz << std::endl;

    p_prom->timestep.set_value();
    delete p_prom;
}
#endif

// Mocking CPU thread work
void do_step_thread (double sleep, double time_zero, std::promise<void> timestep)
{
    amrex::Print() << "Beginning thread work at " << amrex::second() - time_zero << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending thread work at " << amrex::second() - time_zero << std::endl;

    timestep.set_value();
}

// Mocking primary thread work
void do_step (double sleep, double time_zero)
{
    amrex::Print() << "Beginning host work at " << amrex::second() - time_zero << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending host work at " << amrex::second() - time_zero << std::endl;
}


int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";

        int max_step = 10;
        int timestep = 0;

        double sleep = 0.5;
        double sleep_thread = 1.0;
        double time_zero = amrex::second();

        for (; timestep < max_step; timestep++)
        {
           amrex::Print() << "======================================" << std::endl;
           amrex::Print() << "Starting timestep " << timestep
                          << " at " << amrex::second() - time_zero << std::endl;

           std::promise<void> timestep_promise;
           std::future<void> timestep_future = timestep_promise.get_future();
#ifdef AMREX_USE_CUDA
           auto p = new gpu_promise({sleep_thread, time_zero, std::move(timestep_promise)});
           AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu,
                                                   (void*) p ));
#else
           std::thread thread(do_step_thread, sleep_thread, time_zero,
                              std::move(timestep_promise));
#endif
           do_step(sleep, time_zero); 

           amrex::Print() << "Waiting for timestep " << timestep
                          << " to finish at " << amrex::second() - time_zero << std::endl;

           timestep_future.wait();

           amrex::Print() << "Finishing timestep " << timestep
                          << " at " << amrex::second() - time_zero << std::endl;

#ifndef AMREX_USE_CUDA
           thread.join();
#endif
        }

        amrex::Print() << "======================================" << std::endl;
    }
    amrex::Finalize();
}

