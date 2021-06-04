#ifndef AMINI_CPC_H
#define AMINI_CPC_H

#include <memory>
#include <AMini_Defs.H>
#include <AMini_FBI.H>

struct CommMetaData
{
    // The cache of local and send/recv per FillBoundary() or ParallelCopy().
    bool m_threadsafe_loc = false;
    bool m_threadsafe_rcv = false;
    std::unique_ptr<CopyComTagsContainer>      m_LocTags;
    std::unique_ptr<MapOfCopyComTagContainers> m_SndTags;
    std::unique_ptr<MapOfCopyComTagContainers> m_RcvTags;
};

struct CPC
    : CommMetaData
{
    CPC (const amrex::FabArrayBase& dstfa, const amrex::IntVect& dstng,
         const amrex::FabArrayBase& srcfa, const amrex::IntVect& srcng,
         const amrex::Periodicity& period);
    CPC (const amrex::BoxArray& dstba, const amrex::DistributionMapping& dstdm,
         const std::vector<int>& dstidx, const amrex::IntVect& dstng,
         const amrex::BoxArray& srcba, const amrex::DistributionMapping& srcdm,
         const std::vector<int>& srcidx, const amrex::IntVect& srcng,
         const amrex::Periodicity& period, int myproc);
    CPC (const amrex::BoxArray& ba, const amrex::IntVect& ng,
         const amrex::DistributionMapping& dstdm, const amrex::DistributionMapping& srcdm);
    ~CPC ();

    long bytes () const;

    amrex::IntVect     m_srcng;
    amrex::IntVect     m_dstng;
    amrex::Periodicity m_period;
    amrex::BoxArray    m_srcba;
    amrex::BoxArray    m_dstba;
    //
    long        m_nuse;

    private:
    void define (const amrex::BoxArray& ba_dst, const amrex::DistributionMapping& dm_dst,
                 const std::vector<int>& imap_dst,
                 const amrex::BoxArray& ba_src, const amrex::DistributionMapping& dm_src,
                 const std::vector<int>& imap_src,
                 int MyProc = amrex::ParallelDescriptor::MyProc());
};

template <class FAB>
struct PCData {

    const CPC*                  cpc = nullptr;
    const amrex::FabArray<FAB>* src = nullptr;
    amrex::FabArray<FAB>*       dst = nullptr;
    CpOp                        op;
    int                         tag = -1;
    int                         actual_n_rcvs = -1;
    amrex::IntVect              snghost, dnghost;
    amrex::Periodicity          period;
    int                         SC = -1, NC = -1, DC = -1;

    char*                      the_recv_data = nullptr;
    char*                      the_send_data = nullptr;
    std::vector<int>         recv_from;
    std::vector<char*>       recv_data;
    std::vector<std::size_t> recv_size;
    amrex::Vector<MPI_Request> recv_reqs;
    amrex::Vector<MPI_Request> send_reqs;

};

#endif
