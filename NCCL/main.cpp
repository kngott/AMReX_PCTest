#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Arena.H>
#include <AMReX_Random.H>
//#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>

#include <nccl.h>

using namespace amrex;
void main_main ();

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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

    int n_warmup, n_tests, check, do_aware;
    int min_elements, max_elements, factor;
    Real epsilon;
    {
        ParmParse pp;
        pp.get("warmup_count", n_warmup);
        pp.get("test_count", n_tests);

        pp.get("min_elements", min_elements);
        pp.get("max_elements", max_elements);
        pp.get("mult_factor", factor);

        pp.get("check_result", check);
        pp.get("epsilon", epsilon);
        pp.get("do_cuda_aware_mpi", do_aware);
    }

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************

    // NCCL Comm Setup
    ncclComm_t nccl_comm;
    {
        ncclUniqueId id;

        int nRanks = ParallelDescriptor::NProcs();
        int myProc = ParallelDescriptor::MyProc();

        // get NCCL unique ID at rank 0 and broadcast it to all others
        if (myProc == 0) NCCLCHECK(ncclGetUniqueId(&id));
        ParallelDescriptor::Bcast((char*) (&id), sizeof(id));

        // initializing NCCL
        NCCLCHECK(ncclCommInitRank(&nccl_comm, nRanks, id, myProc));
    }

    ncclDataType_t NCCLTYPE;
    if (sizeof(Real) == sizeof(float)) {
        NCCLTYPE = ncclFloat;
    } else if (sizeof(Real) == sizeof(double)) {
        NCCLTYPE = ncclDouble;
    }

    MPI_Comm comm = ParallelDescriptor::Communicator();

    // NCCL API calls:
    // Allreduce, Bcast, Reduce, Allgather, ReduceScatter

    for (int n_ele=min_elements; n_ele<=max_elements; n_ele*=factor) {

        BL_PROFILE_REGION("Test = " + std::to_string(n_ele));

        size_t sz = sizeof(Real)*n_ele;

        void* c_buff = The_Cpu_Arena()->alloc(sz);
        void* p_buff = The_Pinned_Arena()->alloc(sz);

        void* d1_buff = The_Device_Arena()->alloc(sz);
        void* d2_buff = The_Device_Arena()->alloc(sz);
        void* d3_buff = The_Device_Arena()->alloc(sz);
        void* d4_buff = The_Device_Arena()->alloc(sz);

        std::vector<Real> data(n_ele, 0);
        std::vector<Real> cpu(n_ele, 0);

        for (auto& i : data) { i = RandomNormal(1.0, 0.5); }

        // CPU to CPU Initial test
        for (int i=0; i<n_warmup+n_tests; ++i)
        {
            cpu = data;

            if (i >= n_warmup) {
                BL_PROFILE("amrex::AllReduce(): CPU only - " + std::to_string(n_ele));
            }
            amrex::ParallelAllReduce::Sum<Real>(cpu.data(), n_ele, comm);
        }

        // Current Method: GPU->CPU->comm-on-cpu->GPU
        for (int i=0; i<n_warmup+n_tests; ++i)
        {
            Gpu::htod_memcpy(d1_buff, data.data(), sz);
            {
                if (i >= n_warmup) {
                    BL_PROFILE("amrex::AllReduce(): GPU to GPU - " + std::to_string(n_ele));
                }
                Gpu::dtoh_memcpy(p_buff, d1_buff, sz);
                amrex::ParallelAllReduce::Sum<Real>(reinterpret_cast<Real*>(p_buff), n_ele, comm);
                Gpu::htod_memcpy(d1_buff, p_buff, sz);
            }
        }

        // CUDA Aware MPI
        if (do_aware) {
            for (int i=0; i<n_warmup+n_tests; ++i)
            {
                Gpu::htod_memcpy(d2_buff, data.data(), sz);
                {
                    if (i >= n_warmup) {
                        BL_PROFILE("amrex::AllReduce(): CUDA Aware - " + std::to_string(n_ele));
                    }
                    amrex::ParallelAllReduce::Sum<Real>(reinterpret_cast<Real*>(d2_buff), n_ele, comm);
                }
            }
        }
    

        // NCCL: Device-to-device
        for (int i=0; i<n_warmup+n_tests; ++i)
        {
            Gpu::htod_memcpy(d3_buff, data.data(), sz);
            {
                if (i >= n_warmup) {
                    BL_PROFILE("amrex::AllReduce(): NCCL - " + std::to_string(n_ele));
                }
 
                NCCLCHECK( ncclAllReduce(d3_buff, d4_buff, n_ele,
                                         NCCLTYPE, ncclSum, nccl_comm, Gpu::Device::gpuStream()) );

                Gpu::Device::synchronize();
            }
        }

        if (check) {

            std::vector<Real> gpu(n_ele), nccl(n_ele), aware(n_ele);

            // ---------
            // Get data

            Gpu::dtoh_memcpy(c_buff, d1_buff, sz);
            for (int i=0; i<n_ele; ++i)
                { gpu[i] = reinterpret_cast<Real*>(c_buff)[i]; }

            Gpu::dtoh_memcpy(c_buff, d4_buff, sz);
            for (int i=0; i<n_ele; ++i)
                { nccl[i] = reinterpret_cast<Real*>(c_buff)[i]; }

            if (do_aware) {
                Gpu::dtoh_memcpy(c_buff, d2_buff, sz);
                for (int i=0; i<n_ele; ++i)
                    { aware[i] = reinterpret_cast<Real*>(c_buff)[i]; }
            }

            // ----------

            int wrong = 0;
            for (int i=0; i<n_ele; ++i) {
                double diff = std::max(std::abs(cpu[i]-gpu[i]), std::abs(gpu[i]-nccl[i]));
                if (do_aware) {
                    diff = std::max(diff, std::abs(cpu[i]-aware[i]));
                }

                if ( diff >= epsilon ) {
                    wrong++;
                    amrex::Print() << i << " doesn't match: "
                                   << std::setprecision(17) << cpu[i]
                                                     << " " << gpu[i]
                                                     << " " << nccl[i];
                    if (do_aware) {
                        amrex::Print() << " " << std::setprecision(17) << aware[i] << std::endl;
                    }
                }
            }

            if (wrong == 0)
                { amrex::Print() << "All reductions match!" << std::endl; }
        }

        The_Cpu_Arena()->free(c_buff);
        The_Pinned_Arena()->free(p_buff);
        The_Device_Arena()->free(d1_buff);
        The_Device_Arena()->free(d2_buff);
        The_Device_Arena()->free(d3_buff);
        The_Device_Arena()->free(d4_buff);

    }

    NCCLCHECK(ncclCommDestroy(nccl_comm));
}
