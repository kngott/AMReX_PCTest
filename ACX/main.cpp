#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

using namespace amrex;
void main_main ();

#ifndef AMREX_USE_CUDA
#error MPI-ACX is CUDA only.
#endif

struct timers {
    int num;
    double zero;
    double start;
};

#ifdef AMREX_USE_CUDA
// Mocking GPU callback function work
void CUDART_CB print_time(void* p)
{
    auto clock = reinterpret_cast<timers*>(p);

    double end = amrex::second();

    amrex::Print() << "FB #" << clock->num << " completed = " << double(end - clock->start)
                   << "\taverage: " << double(end-clock->zero)/(clock->num+1)
                   << "\tclock: "   << double(end-clock->zero) << std::endl;

    std::free(clock);
}
#endif

// ================================================

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}

void main_main ()
{

    BL_PROFILE("main");

    int ncomp, steps, sync_steps;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);

        pp.get("steps", steps);
    }

    sync_steps = 10;

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************
    {
        Box domain(IntVect{0}, d_size-1);
        BoxArray ba(domain);
        ba.maxSize(mgs);

        Periodicity period(piv);
        DistributionMapping dm(ba);

        MultiFab mf;
        mf.define(ba, dm, ncomp, nghost);
        mf.setVal(3.14159);

        double start_time = amrex::second(); 

        ParallelDescriptor::Barrier();

        for (int i=0; i<5; ++i)
        {
            double time = amrex::second();

            mf.FillBoundary(period);
            amrex::ParallelDescriptor::Barrier();

            double end_time = amrex::second();

            amrex::Print() << "FB init # " << i << " = " << double(end_time - time)
                           << "\taverage: " << double(end_time-start_time)/(i+1)
                           << "\tclock: " << double(end_time - start_time) << std::endl;
        }

        amrex::Print() << "*****************************" << std::endl;

        start_time = amrex::second();

        // ======================================================

        for (int i=0; i<steps; ++i)
        {
            double time = amrex::second();

            mf.FillBoundary(period);
            amrex::ParallelDescriptor::Barrier();

            double end_time = amrex::second();

            amrex::Print() << "FB # " << i << " = " << double(end_time - time)
                           << "\taverage: " << double(end_time-start_time)/(i+1)
                           << "\tclock: " << double(end_time - start_time) << std::endl;
        }

        amrex::Print() << "*****************************" << std::endl;

        start_time = amrex::second();

        for (int i=0; i<steps; ++i)
        {
            timers* t = reinterpret_cast<timers*>(std::malloc(sizeof(timers)));
            double time = amrex::second();
            amrex::Print() << "FB # " << i << " = " << " started: " << double(time - start_time) << std::endl;

            mf.FillBoundary(period, false, true);

            double thread_time = amrex::second();
            amrex::Print() << "FB # " << i << " = " << " launched: " << double(thread_time - time)
                           << "\taverage: " << double(thread_time-start_time)/(i+1)
                           << "\tclock: " << double(thread_time - start_time) << std::endl;
            t->num   = i;
            t->zero  = start_time;
            t->start = time;
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), print_time, (void*) t));

            if ((i+1) % sync_steps == 0) {
                Gpu::synchronize();
                amrex::ParallelDescriptor::Barrier();

                double end_time = amrex::second();

                amrex::Print() << "Group # "    << (i+1)/sync_steps << " = " << double(end_time - time)
                               << "\taverage: " << double(end_time-start_time)/(i+1) << std::endl
                               << "\tclock: "   << double(end_time - start_time) << std::endl;

            }
        }

        Gpu::synchronize();
        amrex::ParallelDescriptor::Barrier();
        double end_time = amrex::second();

        amrex::Print() << "Done = " << double(end_time - start_time) << std::endl;
    }
}
