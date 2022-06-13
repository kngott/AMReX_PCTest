#include <AMReX_Config.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Print.H>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>

#if 0
FB define_fb   (const FabArrayBase& fa);
FB define_epo  (const FabArrayBase& fa);
FB define_os   (const FabArrayBase& fa);
FB tag_one_box (int krcv, BoxArray const& ba, DistributionMapping const& dm,
                bool build_recv_tag);
#endif

// To avoid calling expensive define
void setup_cpc (amrex::FabArrayBase::CPC& cpc);

void
define_cpc (amrex::FabArrayBase::CPC& my_cpc,
            const amrex::BoxArray& ba_dst,      const amrex::DistributionMapping& dm_dst,
            const amrex::Vector<int>& imap_dst, const amrex::IntVect& dstng,
            const amrex::BoxArray& ba_src,      const amrex::DistributionMapping& dm_src,
            const amrex::Vector<int>& imap_src, const amrex::IntVect& srcng,
            const amrex::Periodicity& period,   bool to_ghost_cells_only, int MyProc);
