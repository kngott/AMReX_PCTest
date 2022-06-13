#include <AMReX_FabArrayBase.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_NonLocalBC.H>

#include <AMReX_BArena.H>
#include <AMReX_CArena.H>

void
define_cpc (amrex::FabArrayBase::CPC& my_cpc,
            const amrex::BoxArray& ba_dst,      const amrex::DistributionMapping& dm_dst,
            const amrex::Vector<int>& imap_dst, const amrex::IntVect& dstng,
            const amrex::BoxArray& ba_src,      const amrex::DistributionMapping& dm_src,
            const amrex::Vector<int>& imap_src, const amrex::IntVect& srcng,
            const amrex::Periodicity& period,   bool to_ghost_cells_only, int MyProc)
{

    my_cpc.m_srcng  = srcng;
    my_cpc.m_dstng  = dstng;
    my_cpc.m_period = period;
    my_cpc.m_tgco   = to_ghost_cells_only;
    my_cpc.m_srcba  = ba_src;
    my_cpc.m_dstba  = ba_dst;
    my_cpc.m_nuse   = 0;

    using CopyComTag = amrex::FabArrayBase::CopyComTag;

    BL_ASSERT(ba_dst.size() > 0 && ba_src.size() > 0);
    BL_ASSERT(ba_dst.ixType() == ba_src.ixType());

    my_cpc.m_LocTags = std::make_unique<CopyComTag::CopyComTagsContainer>();
    my_cpc.m_SndTags = std::make_unique<CopyComTag::MapOfCopyComTagContainers>();
    my_cpc.m_RcvTags = std::make_unique<CopyComTag::MapOfCopyComTagContainers>();

    if (!(imap_dst.empty() && imap_src.empty()))
    {
        const int nlocal_src = imap_src.size();
        const amrex::IntVect& ng_src = my_cpc.m_srcng;
        const int nlocal_dst = imap_dst.size();
        const amrex::IntVect& ng_dst = my_cpc.m_dstng;

        std::vector< std::pair<int,amrex::Box> > isects;

        const std::vector<amrex::IntVect>& pshifts = my_cpc.m_period.shiftIntVect();

        auto& send_tags = *my_cpc.m_SndTags;

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
//                    if (amrex::ParallelDescriptor::sameTeam(dst_owner))
                    if (MyProc == dst_owner) {
                        continue; // local copy will be dealt with later
                    } else if (MyProc == dm_src[k_src]) {
                        amrex::BoxList const bl_dst = my_cpc.m_tgco ? boxDiff(bx, ba_dst[k_dst]) : amrex:: BoxList(bx);
                        for (auto const& b : bl_dst) {
                            send_tags[dst_owner].push_back(CopyComTag(b, b-(*pit), k_dst, k_src));
                        }
                    }
                }
            }
        }

        auto& recv_tags = *my_cpc.m_RcvTags;

        amrex::BoxList bl_local(ba_dst.ixType());
        amrex::BoxList bl_remote(ba_dst.ixType());

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

//        if (amrex::ParallelDescriptor::TeamSize() > 1)
        if (false) { // For now, TeamSize == 1.
                   // Team math relys on ParallelDescriptor::MyProc(), so would need editing here.
            check_local = true;
        }

        for (int i = 0; i < nlocal_dst; ++i)
        {
            const int   k_dst = imap_dst[i];
            const amrex::Box& bx_dst_valid = ba_dst[k_dst];
            const amrex::Box& bx_dst = amrex::grow(bx_dst_valid, ng_dst);

            for (std::vector<amrex::IntVect>::const_iterator pit=pshifts.begin(); pit!=pshifts.end(); ++pit)
            {
                ba_src.intersections(bx_dst+(*pit), isects, false, ng_src);

                for (int j = 0, M = isects.size(); j < M; ++j)
                {
                    const int k_src     = isects[j].first;
                    const amrex::Box& bx       = isects[j].second - *pit;
                    const int src_owner = dm_src[k_src];

                    amrex::BoxList const bl_dst = my_cpc.m_tgco ? boxDiff(bx,bx_dst_valid) : amrex::BoxList(bx);
                    for (auto const& b : bl_dst) {
 //                    if (amrex::ParallelDescriptor::sameTeam(src_owner, MyProc)) // local copy
                       if (MyProc == src_owner) {
//                        if (amrex::ParallelDescriptor::sameTeam(src_owner, MyProc)) // local copy
                            const amrex::BoxList tilelist(b, amrex::FabArrayBase::comm_tile_size);
                            for (auto const& btile : tilelist) {
                                my_cpc.m_LocTags->push_back(CopyComTag(btile, btile+(*pit), k_dst, k_src));
                            }
                            if (check_local) {
                                bl_local.push_back(b);
                            }
                        } else if (MyProc == dm_dst[k_dst]) {
                            recv_tags[src_owner].push_back(CopyComTag(b, b+(*pit), k_dst, k_src));
                            if (check_remote) {
                                bl_remote.push_back(b);
                            }
                        }
                    }
                }
            }
        }

        if (bl_local.size() <= 1) {
            my_cpc.m_threadsafe_loc = true;
        } else {
            my_cpc.m_threadsafe_loc = amrex::BoxArray(std::move(bl_local)).isDisjoint();
        }

        if (bl_remote.size() <= 1) {
            my_cpc.m_threadsafe_rcv = true;
        } else {
            my_cpc.m_threadsafe_rcv = amrex::BoxArray(std::move(bl_remote)).isDisjoint();
        }

        for (int ipass = 0; ipass < 2; ++ipass) // pass 0: send; pass 1: recv
        {
            CopyComTag::MapOfCopyComTagContainers & Tags = (ipass == 0) ? *my_cpc.m_SndTags : *my_cpc.m_RcvTags;
            for (auto& kv : Tags)
            {
                std::vector<CopyComTag>& cctv = kv.second;
                // We need to fix the order so that the send and recv processes match.
                std::sort(cctv.begin(), cctv.end());
            }
        }
    }
}

#if 0

void
FabArrayBase::FB::define_fb (const FabArrayBase& fa)
{
    AMREX_ASSERT(m_multi_ghost ? fa.nGrow() >= 2 : true); // must have >= 2 ghost nodes
    AMREX_ASSERT(m_multi_ghost ? !m_period.isAnyPeriodic() : true); // this only works for non-periodic
    const int                  MyProc   = ParallelDescriptor::MyProc();
    const BoxArray&            ba       = fa.boxArray();
    const DistributionMapping& dm       = fa.DistributionMap();
    const Vector<int>&         imap     = fa.IndexArray();

    // For local copy, all workers in the same team will have the identical copy of tags
    // so that they can share work.  But for remote communication, they are all different.

    const int nlocal = imap.size();
    const IntVect& ng = m_ngrow;
    const IntVect ng_ng = m_ngrow - 1;
    std::vector< std::pair<int,Box> > isects;

    const std::vector<IntVect>& pshifts = m_period.shiftIntVect();

    auto& send_tags = *m_SndTags;

    for (int i = 0; i < nlocal; ++i)
    {
        const int ksnd = imap[i];
        const Box& vbx = ba[ksnd];
        const Box& vbx_ng  = amrex::grow(vbx,1);

        for (auto pit=pshifts.cbegin(); pit!=pshifts.cend(); ++pit)
        {
            ba.intersections(vbx+(*pit), isects, false, ng);

            for (int j = 0, M = isects.size(); j < M; ++j)
            {
                const int krcv      = isects[j].first;
                const Box& bx       = isects[j].second;
                const int dst_owner = dm[krcv];

                if (ParallelDescriptor::sameTeam(dst_owner)) {
                    continue;  // local copy will be dealt with later
                } else if (MyProc == dm[ksnd]) {
                    BoxList bl = amrex::boxDiff(bx, ba[krcv]);
                    if (m_multi_ghost)
                    {
                        // In the case where ngrow>1, augment the send/rcv box list
                        // with boxes for overlapping ghost nodes.
                        const Box& ba_krcv   = amrex::grow(ba[krcv],1);
                        const Box& dst_bx_ng = (amrex::grow(ba_krcv,ng_ng) & (vbx_ng + (*pit)));
                        const BoxList &bltmp = ba.complementIn(dst_bx_ng);
                        for (auto const& btmp : bltmp)
                        {
                            bl.join(amrex::boxDiff(btmp,ba_krcv));
                        }
                        bl.simplify();
                    }
                    for (BoxList::const_iterator lit = bl.begin(); lit != bl.end(); ++lit)
                        send_tags[dst_owner].push_back(CopyComTag(*lit, (*lit)-(*pit), krcv, ksnd));
                }
            }
        }
    }

    auto& recv_tags = *m_RcvTags;

    BoxList bl_local(ba.ixType());
    BoxList bl_remote(ba.ixType());

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

    if (ParallelDescriptor::TeamSize() > 1) {
        check_local = true;
    }

    for (int i = 0; i < nlocal; ++i)
    {
        const int   krcv = imap[i];
        const Box& vbx   = ba[krcv];
        const Box& vbx_ng  = amrex::grow(vbx,1);
        const Box& bxrcv = amrex::grow(vbx, ng);

        for (auto pit=pshifts.cbegin(); pit!=pshifts.cend(); ++pit)
        {
            ba.intersections(bxrcv+(*pit), isects);

            for (int j = 0, M = isects.size(); j < M; ++j)
            {
                const int ksnd      = isects[j].first;
                const Box& dst_bx   = isects[j].second - *pit;
                const int src_owner = dm[ksnd];

                BoxList bl = amrex::boxDiff(dst_bx, vbx);

                if (m_multi_ghost)
                {
                    // In the case where ngrow>1, augment the send/rcv box list
                    // with boxes for overlapping ghost nodes.
                    Box ba_ksnd = ba[ksnd];
                    ba_ksnd.grow(1);
                    const Box dst_bx_ng = (ba_ksnd & (bxrcv + (*pit))) - (*pit);
                    const BoxList &bltmp = ba.complementIn(dst_bx_ng);
                    for (auto const& btmp : bltmp)
                    {
                        bl.join(amrex::boxDiff(btmp,vbx_ng));
                    }
                    bl.simplify();
                }
                for (BoxList::const_iterator lit = bl.begin(); lit != bl.end(); ++lit)
                {
                    const Box& blbx = *lit;

                    if (ParallelDescriptor::sameTeam(src_owner)) { // local copy
                        const BoxList tilelist(blbx, FabArrayBase::comm_tile_size);
                        for (BoxList::const_iterator
                                 it_tile  = tilelist.begin(),
                                 End_tile = tilelist.end();   it_tile != End_tile; ++it_tile)
                        {
                            m_LocTags->push_back(CopyComTag(*it_tile, (*it_tile)+(*pit), krcv, ksnd));
                        }
                        if (check_local) {
                            bl_local.push_back(blbx);
                        }
                    } else if (MyProc == dm[krcv]) {
                        recv_tags[src_owner].push_back(CopyComTag(blbx, blbx+(*pit), krcv, ksnd));
                        if (check_remote) {
                            bl_remote.push_back(blbx);
                        }
                    }
                }
            }
        }

        if (bl_local.size() <= 1) {
            m_threadsafe_loc = true;
        } else {
            m_threadsafe_loc = BoxArray(std::move(bl_local)).isDisjoint();
        }

        if (bl_remote.size() <= 1) {
            m_threadsafe_rcv = true;
        } else {
            m_threadsafe_rcv = BoxArray(std::move(bl_remote)).isDisjoint();
        }
    }

    for (int ipass = 0; ipass < 2; ++ipass) // pass 0: send; pass 1: recv
    {
        CopyComTag::MapOfCopyComTagContainers & Tags = (ipass == 0) ? *m_SndTags : *m_RcvTags;

        for (auto& kv : Tags)
        {
            std::vector<CopyComTag>& cctv = kv.second;

            // We need to fix the order so that the send and recv processes match.
            std::sort(cctv.begin(), cctv.end());

            std::vector<CopyComTag> cctv_tags_cross;
            cctv_tags_cross.reserve(cctv.size());

            for (auto const& tag : cctv)
            {
                const Box& bx = tag.dbox;
                const IntVect& d2s = tag.sbox.smallEnd() - tag.dbox.smallEnd();

                std::vector<Box> boxes;
                if (m_cross) {
                    const Box& dstvbx = ba[tag.dstIndex];
                    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                    {
                        Box lo = dstvbx;
                        lo.setSmall(dir, dstvbx.smallEnd(dir) - ng[dir]);
                        lo.setBig  (dir, dstvbx.smallEnd(dir) - 1);
                        lo &= bx;
                        if (lo.ok()) {
                            boxes.push_back(lo);
                        }

                        Box hi = dstvbx;
                        hi.setSmall(dir, dstvbx.bigEnd(dir) + 1);
                        hi.setBig  (dir, dstvbx.bigEnd(dir) + ng[dir]);
                        hi &= bx;
                        if (hi.ok()) {
                            boxes.push_back(hi);
                        }
                    }
                } else {
                    boxes.push_back(bx);
                }

                if (!boxes.empty())
                {
                    for (auto const& cross_box : boxes)
                    {
                        if (m_cross)
                        {
                            cctv_tags_cross.push_back(CopyComTag(cross_box, cross_box+d2s,
                                                                 tag.dstIndex, tag.srcIndex));
                        }
                    }
                }
            }

            if (!cctv_tags_cross.empty()) {
                cctv.swap(cctv_tags_cross);
            }
        }
    }
}

void
FabArrayBase::FB::define_epo (const FabArrayBase& fa)
{
    const int                  MyProc   = ParallelDescriptor::MyProc();
    const BoxArray&            ba       = fa.boxArray();
    const DistributionMapping& dm       = fa.DistributionMap();
    const Vector<int>&         imap     = fa.IndexArray();

    // For local copy, all workers in the same team will have the identical copy of tags
    // so that they can share work.  But for remote communication, they are all different.

    const int nlocal = imap.size();
    const IntVect& ng = m_ngrow;
    const IndexType& typ = ba.ixType();
    std::vector< std::pair<int,Box> > isects;

    const std::vector<IntVect>& pshifts = m_period.shiftIntVect();

    auto& send_tags = *m_SndTags;

    Box pdomain = m_period.Domain();
    pdomain.convert(typ);

    for (int i = 0; i < nlocal; ++i)
    {
        const int ksnd = imap[i];
        Box bxsnd = amrex::grow(ba[ksnd],ng);
        bxsnd &= pdomain; // source must be inside the periodic domain.

        if (!bxsnd.ok()) continue;

        for (auto pit=pshifts.cbegin(); pit!=pshifts.cend(); ++pit)
        {
            if (*pit != IntVect::TheZeroVector())
            {
                ba.intersections(bxsnd+(*pit), isects, false, ng);

                for (int j = 0, M = isects.size(); j < M; ++j)
                {
                    const int krcv      = isects[j].first;
                    const Box& bx       = isects[j].second;
                    const int dst_owner = dm[krcv];

                    if (ParallelDescriptor::sameTeam(dst_owner)) {
                        continue;  // local copy will be dealt with later
                    } else if (MyProc == dm[ksnd]) {
                        const BoxList& bl = amrex::boxDiff(bx, pdomain);
                        for (BoxList::const_iterator lit = bl.begin(); lit != bl.end(); ++lit) {
                            send_tags[dst_owner].push_back(CopyComTag(*lit, (*lit)-(*pit), krcv, ksnd));
                        }
                    }
                }
            }
        }
    }

    auto& recv_tags = *m_RcvTags;

    BoxList bl_local(ba.ixType());
    BoxList bl_remote(ba.ixType());

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

    if (ParallelDescriptor::TeamSize() > 1) {
        check_local = true;
    }

    for (int i = 0; i < nlocal; ++i)
    {
        const int   krcv = imap[i];
        const Box& vbx   = ba[krcv];
        const Box& bxrcv = amrex::grow(vbx, ng);

        if (pdomain.contains(bxrcv)) continue;

        for (std::vector<IntVect>::const_iterator pit=pshifts.begin(); pit!=pshifts.end(); ++pit)
        {
            if (*pit != IntVect::TheZeroVector())
            {
                ba.intersections(bxrcv+(*pit), isects, false, ng);

                for (int j = 0, M = isects.size(); j < M; ++j)
                {
                    const int ksnd      = isects[j].first;
                    const Box& dst_bx   = isects[j].second - *pit;
                    const int src_owner = dm[ksnd];

                    const BoxList& bl = amrex::boxDiff(dst_bx, pdomain);

                    for (BoxList::const_iterator lit = bl.begin(); lit != bl.end(); ++lit)
                    {
                        Box sbx = (*lit) + (*pit);
                        sbx &= pdomain; // source must be inside the periodic domain.

                        if (sbx.ok()) {
                            Box dbx = sbx - (*pit);
                            if (ParallelDescriptor::sameTeam(src_owner)) { // local copy
                                const BoxList tilelist(dbx, FabArrayBase::comm_tile_size);
                                for (BoxList::const_iterator
                                         it_tile  = tilelist.begin(),
                                         End_tile = tilelist.end();   it_tile != End_tile; ++it_tile)
                                {
                                    m_LocTags->push_back(CopyComTag(*it_tile, (*it_tile)+(*pit), krcv, ksnd));
                                }
                                if (check_local) {
                                    bl_local.push_back(dbx);
                                }
                            } else if (MyProc == dm[krcv]) {
                                recv_tags[src_owner].push_back(CopyComTag(dbx, sbx, krcv, ksnd));
                                if (check_remote) {
                                    bl_remote.push_back(dbx);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (bl_local.size() <= 1) {
            m_threadsafe_loc = true;
        } else {
            m_threadsafe_loc = BoxArray(std::move(bl_local)).isDisjoint();
        }

        if (bl_remote.size() <= 1) {
            m_threadsafe_rcv = true;
        } else {
            m_threadsafe_rcv = BoxArray(std::move(bl_remote)).isDisjoint();
        }
    }

    for (int ipass = 0; ipass < 2; ++ipass) // pass 0: send; pass 1: recv
    {
        CopyComTag::MapOfCopyComTagContainers & Tags = (ipass == 0) ? *m_SndTags : *m_RcvTags;
        for (auto& kv : Tags)
        {
            std::vector<CopyComTag>& cctv = kv.second;
            // We need to fix the order so that the send and recv processes match.
            std::sort(cctv.begin(), cctv.end());
        }
    }
}

void FabArrayBase::FB::tag_one_box (int krcv, BoxArray const& ba, DistributionMapping const& dm,
                                    bool build_recv_tag)
{
    Box const& vbx = ba[krcv];
    Box const& gbx = amrex::grow(vbx, m_ngrow);
    IndexType const ixtype = vbx.ixType();

    std::vector<std::pair<int,Box> > isects2;
    std::vector<std::tuple<int,Box,IntVect> > isects3;
    auto const& pshifts = m_period.shiftIntVect();
    for (auto const& shft: pshifts) {
        ba.intersections(gbx+shft, isects2);
        for (auto const& is2 : isects2) {
            if (is2.first != krcv || shft != 0) {
                isects3.emplace_back(is2.first, is2.second-shft, shft);
            }
        }
    }

    int const dst_owner = dm[krcv];
    bool const is_receiver = dst_owner == ParallelDescriptor::MyProc();

    BoxList bl(ixtype);
    BoxList tmpbl(ixtype);
    for (auto const& is3 : isects3) {
        int const      ksnd = std::get<int>(is3);
        Box const&   dst_bx = std::get<Box>(is3);
        IntVect const& shft = std::get<IntVect>(is3); // src = dst + shft
        int const src_owner = dm[ksnd];
        bool is_sender = src_owner == ParallelDescriptor::MyProc();

        if ((build_recv_tag && (ParallelDescriptor::sameTeam(src_owner) || is_receiver))
            || (is_sender && !ParallelDescriptor::sameTeam(dst_owner)))
        {
            bl.clear();
            tmpbl.clear();

            if (ksnd < krcv || (ksnd == krcv && shft < IntVect::TheZeroVector())) {
                bl.push_back(dst_bx); // valid cells are allowed to override valid cells
            } else {
                bl = boxDiff(dst_bx, vbx); // exclude valid cells
            }

            for (auto const& o_is3 : isects3) {
                int const      o_ksnd = std::get<int>(o_is3);
                IntVect const& o_shft = std::get<IntVect>(o_is3);
                Box const&   o_dst_bx = std::get<Box>(o_is3);
                if ((o_ksnd < ksnd || (o_ksnd == ksnd && o_shft < shft))
                    && o_dst_bx.intersects(dst_bx))
                {
                    for (auto const& b : bl) {
                        tmpbl.join(boxDiff(b, o_dst_bx));
                    }
                    std::swap(bl, tmpbl);
                    tmpbl.clear();
                }
            }

            if (build_recv_tag) {
                if (ParallelDescriptor::sameTeam(src_owner)) { // local copy
                    for (auto const& b : bl) {
                        const BoxList tilelist(b, FabArrayBase::comm_tile_size);
                        for (auto const& tbx : tilelist) {
                            m_LocTags->emplace_back(tbx, tbx+shft, krcv, ksnd);
                        }
                    }
                } else if (is_receiver) {
                    for (auto const& b : bl) {
                        (*m_RcvTags)[src_owner].emplace_back(b, b+shft, krcv, ksnd);
                    }
                }
            } else if (is_sender && !ParallelDescriptor::sameTeam(dst_owner))  {
                for (auto const& b : bl) {
                    (*m_SndTags)[dst_owner].emplace_back(b, b+shft, krcv, ksnd);
                }
            }
        }
    }


}

void
FabArrayBase::FB::define_os (const FabArrayBase& fa)
{
    m_threadsafe_loc = true;
    m_threadsafe_rcv = true;

    const BoxArray&            ba       = fa.boxArray();
    const DistributionMapping& dm       = fa.DistributionMap();
    const Vector<int>&         imap     = fa.IndexArray();
    const int nlocal = imap.size();

    for (int i = 0; i < nlocal; ++i)
    {
        tag_one_box(imap[i], ba, dm, true);
    }

#ifdef AMREX_USE_MPI
    if (ParallelDescriptor::NProcs() > 1) {
        const std::vector<IntVect>& pshifts = m_period.shiftIntVect();
        std::vector< std::pair<int,Box> > isects;

        std::set<int> my_receiver;
        for (int i = 0; i < nlocal; ++i) {
            int const ksnd = imap[i];
            Box const& vbx = ba[ksnd];
            for (auto const& shft : pshifts) {
                ba.intersections(vbx+shft, isects, false, m_ngrow);
                for (auto const& is : isects) {
                    if (is.first != ksnd || shft != 0) {
                        my_receiver.insert(is.first);
                    }
                }
            }
        }

        // Unlike normal FillBoundary, we have to build the send tags
        // differently.  This is because (b1 \ b2) \ b3 might produce
        // different BoxList than (b1 \ b3) \ b2, not just in the order of
        // Boxes in BoxList that can be fixed by sorting.  To make sure the
        // send tags on the sender process matches the recv tags on the
        // receiver process, we make the sender to use the same procedure to
        // build tags as the receiver.

        for (auto const& krcv : my_receiver) {
            tag_one_box(krcv, ba, dm, false);
        }
    }
#endif

    // No need to sort send and recv tags because they are already sorted
    // due to the way they are built.
}

#endif
