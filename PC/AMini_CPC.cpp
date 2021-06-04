#include <AMini_CPC.H>
#include <AMReX_ParallelDescriptor.H>

CPC::CPC (const amrex::FabArrayBase& dstfa, const amrex::IntVect& dstng,
          const amrex::FabArrayBase& srcfa, const amrex::IntVect& srcng,
          const amrex::Periodicity& period)
: m_srcng(srcng), m_dstng(dstng), m_period(period),
  m_srcba(srcfa.boxArray()), m_dstba(dstfa.boxArray()), m_nuse(0)
{
    this->define(m_dstba, dstfa.DistributionMap(), dstfa.IndexArray(),
                 m_srcba, srcfa.DistributionMap(), srcfa.IndexArray());
}

CPC::CPC (const amrex::BoxArray& dstba, const amrex::DistributionMapping& dstdm,
          const std::vector<int>& dstidx, const amrex::IntVect& dstng,
          const amrex::BoxArray& srcba, const amrex::DistributionMapping& srcdm,
          const std::vector<int>& srcidx, const amrex::IntVect& srcng,
          const amrex::Periodicity& period, int myproc)
: m_srcng(srcng), m_dstng(dstng), m_period(period),
  m_srcba(srcba), m_dstba(dstba), m_nuse(0)
{
    this->define(dstba, dstdm, dstidx, srcba, srcdm, srcidx, myproc);
}

CPC::~CPC ()
{}

void
CPC::define (const amrex::BoxArray& ba_dst,
             const amrex::DistributionMapping& dm_dst,
             const std::vector<int>& imap_dst,
             const amrex::BoxArray& ba_src,
             const amrex::DistributionMapping& dm_src,
             const std::vector<int>& imap_src,
             int MyProc)
{
    BL_PROFILE("CPC::define()");

    BL_ASSERT(ba_dst.size() > 0 && ba_src.size() > 0);
    BL_ASSERT(ba_dst.ixType() == ba_src.ixType());

    m_LocTags = std::make_unique<CopyComTagsContainer>();
    m_SndTags = std::make_unique<MapOfCopyComTagContainers>();
    m_RcvTags = std::make_unique<MapOfCopyComTagContainers>();

    if (!(imap_dst.empty() && imap_src.empty()))
    {
        const int nlocal_src = imap_src.size();
        const amrex::IntVect& ng_src = m_srcng;
        const int nlocal_dst = imap_dst.size();
        const amrex::IntVect& ng_dst = m_dstng;

        std::vector< std::pair<int,amrex::Box> > isects;

        const std::vector<amrex::IntVect>& pshifts = m_period.shiftIntVect();

        auto& send_tags = *m_SndTags;

        for (int i = 0; i < nlocal_src; ++i)
        {
            const int   k_src = imap_src[i];
            const amrex::Box& bx_src = amrex::grow(ba_src[k_src], ng_src);

            for (std::vector<amrex::IntVect>::const_iterator pit=pshifts.begin(); pit!=pshifts.end(); ++pit)
            {
                ba_dst.intersections(bx_src+(*pit), isects, false, ng_dst);

                for (int j = 0, M = isects.size(); j < M; ++j)
                {
                    const int k_dst     = isects[j].first;
                    const amrex::Box& bx       = isects[j].second;
                    const int dst_owner = dm_dst[k_dst];

                    if (amrex::ParallelDescriptor::sameTeam(dst_owner)) {
                        continue; // local copy will be dealt with later
                    } else if (MyProc == dm_src[k_src]) {
                        send_tags[dst_owner].push_back(CopyComTag(bx, bx-(*pit), k_dst, k_src));
                    }
                }
            }
        }

        auto& recv_tags = *m_RcvTags;

        amrex::BaseFab<int> localtouch(amrex::The_Cpu_Arena()), remotetouch(amrex::The_Cpu_Arena());
        bool check_local = false, check_remote = false;
#if defined(AMREX_USE_GPU)
        check_local = true;
        check_remote = true;
#elif defined(AMREX_USE_OMP)
        if (omp_get_max_threads() > 1) {
            check_local = true;
            check_remote = true;
        }
#endif

        if (amrex::ParallelDescriptor::TeamSize() > 1) {
            check_local = true;
        }

        m_threadsafe_loc = ! check_local;
        m_threadsafe_rcv = ! check_remote;

        for (int i = 0; i < nlocal_dst; ++i)
        {
            const int   k_dst = imap_dst[i];
            const amrex::Box& bx_dst = amrex::grow(ba_dst[k_dst], ng_dst);

            if (check_local) {
                localtouch.resize(bx_dst);
                localtouch.setVal<amrex::RunOn::Host>(0);
            }

            if (check_remote) {
                remotetouch.resize(bx_dst);
                remotetouch.setVal<amrex::RunOn::Host>(0);
            }

            for (std::vector<amrex::IntVect>::const_iterator pit=pshifts.begin(); pit!=pshifts.end(); ++pit)
            {
                ba_src.intersections(bx_dst+(*pit), isects, false, ng_src);

                for (int j = 0, M = isects.size(); j < M; ++j)
                {
                    const int k_src     = isects[j].first;
                    const amrex::Box& bx       = isects[j].second - *pit;
                    const int src_owner = dm_src[k_src];

                    if (amrex::ParallelDescriptor::sameTeam(src_owner, MyProc)) { // local copy
                        const amrex::BoxList tilelist(bx, amrex::FabArrayBase::comm_tile_size);
                        for (amrex::BoxList::const_iterator
                                 it_tile  = tilelist.begin(),
                                 End_tile = tilelist.end();   it_tile != End_tile; ++it_tile)
                        {
                            m_LocTags->push_back(CopyComTag(*it_tile, (*it_tile)+(*pit), k_dst, k_src));
                        }
                        if (check_local) {
                            localtouch.plus<amrex::RunOn::Host>(1, bx);
                        }
                    } else if (MyProc == dm_dst[k_dst]) {
                        recv_tags[src_owner].push_back(CopyComTag(bx, bx+(*pit), k_dst, k_src));
                        if (check_remote) {
                            remotetouch.plus<amrex::RunOn::Host>(1, bx);
                        }
                    }
                }
            }

            if (check_local) {
                // safe if a cell is touched no more than once
                // keep checking thread safety if it is safe so far
                check_local = m_threadsafe_loc = localtouch.max<amrex::RunOn::Host>() <= 1;
            }

            if (check_remote) {
                check_remote = m_threadsafe_rcv = remotetouch.max<amrex::RunOn::Host>() <= 1;
            }
        }

        for (int ipass = 0; ipass < 2; ++ipass) // pass 0: send; pass 1: recv
        {
            MapOfCopyComTagContainers & Tags = (ipass == 0) ? *m_SndTags : *m_RcvTags;
            for (auto& kv : Tags)
            {
                std::vector<CopyComTag>& cctv = kv.second;
                // We need to fix the order so that the send and recv processes match.
                std::sort(cctv.begin(), cctv.end());
            }
        }
    }
}

CPC::CPC (const amrex::BoxArray& ba, const amrex::IntVect& ng,
          const amrex::DistributionMapping& dstdm, const amrex::DistributionMapping& srcdm)
: m_srcng(ng), m_dstng(ng), m_period(),
  m_srcba(ba), m_dstba(ba), m_nuse(0)
{
    BL_ASSERT(ba.size() > 0);

    m_LocTags = std::make_unique<CopyComTagsContainer>();
    m_SndTags = std::make_unique<MapOfCopyComTagContainers>();
    m_RcvTags = std::make_unique<MapOfCopyComTagContainers>();

    const int myproc = amrex::ParallelDescriptor::MyProc();

    for (int i = 0, N = ba.size(); i < N; ++i)
    {
        const int src_owner = srcdm[i];
        const int dst_owner = dstdm[i];
        if (src_owner == myproc || dst_owner == myproc)
        {
            const amrex::Box& bx = amrex::grow(ba[i], ng);
            const amrex::BoxList tilelist(bx, amrex::FabArrayBase::comm_tile_size);
            if (src_owner == myproc && dst_owner == myproc)
            {
                for (const amrex::Box& tbx : tilelist)
                {
                    m_LocTags->push_back(CopyComTag(tbx, tbx, i, i));
                }
            }
            else
            {
                auto& Tags = (src_owner == myproc) ? (*m_SndTags)[dst_owner] : (*m_RcvTags)[src_owner];
                Tags.push_back(CopyComTag(bx, bx, i, i));
            }
        }
    }
}
