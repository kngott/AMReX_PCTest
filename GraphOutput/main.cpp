#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_DotGraph.H>

using namespace amrex;
void main_main ();

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

    int ncomp;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
    }

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************
    {
        Box domain(IntVect{0}, d_size);
        BoxArray ba(domain);
        ba.maxSize(mgs);

        int nranks = ParallelDescriptor::NProcs();

        long nboxes = ba.size();
        Vector<int> dst_map(nboxes, 0);
        Vector<Real> weights(nboxes, 0);
        Vector<Long> wgts(nboxes, 0);

        for (int i=0; i<nboxes; ++i)
        {
           dst_map[i] = amrex::Random_int(nranks);
           weights[i] = amrex::RandomNormal(5.0, 2.0);
           wgts[i] = long(weights[i]*1000);
        }

        DistributionMapping dm(dst_map);

        MultiFab mf;
        mf.define(ba, dm, ncomp, nghost);

        Periodicity period(piv);

        // ======================================================

        amrex::dot_graph_raw("test", mf, weights, dm);

        // ======================================================

        const amrex::FabArrayBase::FB& the_fb = mf.getFB(nghost, period);

        amrex::dot_graph_raw("fb", mf, weights, the_fb, ncomp, dm);

//        amrex::dot_graph_raw("data", dm, ba, weights);

//        Real* eff = new Real;

//        DistributionMapping dm_knapsack = dm;
//        dm_knapsack.KnapSackProcessorMap(wgts, nranks, eff, true, 
//                                         std::numeric_limits<int>::max(), false);

//        DistributionMapping dm_sfc = dm;
//        dm_sfc.SFCProcessorMap(nboxes, wgts, nranks);

//        amrex::dot_graph(nranks, dm_sfc, weights, "sfc");
//        amrex::dot_graph_python(nranks, dm_sfc, weights, "sfc_py");

//        amrex::dot_graph("knap", dm_knapsack, weights);
//        amrex::dot_graph_python("knap_py", dm_knapsack, weights);

//        delete eff;
    }

}
