#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Graph.H>

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

    // This test specifically needs 2 ghost cells.
    nghost = {2,2,2};

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************
    {
        Box domain(IntVect{0}, d_size);
        BoxArray ba(domain);
        ba.maxSize(mgs);

        int nranks = ParallelDescriptor::NProcs();

        long nboxes = ba.size();
        Vector<int> dst_map_a(nboxes, 0);
        Vector<int> dst_map_b(nboxes, 0);
        Vector<Real> weights_a(nboxes, 0);
        Vector<Real> weights_b(nboxes, 0);

        // For potentially running a KnapSack or SFC.
        Vector<Long> wgts_a(nboxes, 0);
        Vector<Long> wgts_b(nboxes, 0);

        for (int i=0; i<nboxes; ++i)
        {
           dst_map_a[i] = amrex::Random_int(nranks);
           weights_a[i] = amrex::RandomNormal(5.0, 2.0);
           wgts_a[i] = long(weights_a[i]*1000);

           dst_map_b[i] = amrex::Random_int(nranks);
           weights_b[i] = amrex::RandomNormal(2.0, 0.5);
           wgts_b[i] = long(weights_b[i]*1000);
        }

        DistributionMapping dm_a(dst_map_a);
        DistributionMapping dm_b(dst_map_b);

        MultiFab mf_a, mf_b;
        mf_a.define(ba, dm_a, ncomp, nghost);
        mf_b.define(ba, dm_b, ncomp, nghost);

        Periodicity period(piv);

        // ======================================================

        amrex::Graph test_graph;

        test_graph.addFab(mf_a, "A", weights_a, "A-work", ParallelDescriptor::MyProc());
        test_graph.addFab(mf_a, "A", weights_b, "B-work", ParallelDescriptor::MyProc()-1);

        test_graph.addFab(mf_b, "B", weights_b, "B-work", ParallelDescriptor::MyProc()+1);
        test_graph.addFab(mf_b, "B", weights_a, "A-work", ParallelDescriptor::MyProc());

        // FB 1 on A, FB 2 on B, PC between.
        test_graph.addFillBoundary("FB_1", "A", 1.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_a, IntVect{1,1,1}, period);
        test_graph.addFillBoundary("FB_2", "B", 2.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_b, IntVect{2,2,2}, period);
        test_graph.addParallelCopy("PC", "B", "A", 3.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_b, mf_a);

        // ======================================================

        // print both ways
        test_graph.print("readable.graph");

//        test_graph.print_table("table");

        // assemble than print (fix it)
    }
}
