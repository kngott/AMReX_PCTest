#ifndef AMINI_PACK_H
#define AMINI_PACK_H

#include <AMini_Defs.H>

template <class FAB>
void pack_send_buffer_cpu (amrex::FabArray<FAB> const& src, int scomp, int ncomp,
                           std::vector<char*> const& send_data,
                           std::vector<std::size_t> const& send_size,
                           std::vector<CopyComTagsContainer const*> const& send_cctc)
{
    amrex::ignore_unused(send_size);

    const int N_snds = send_data.size();
    if (N_snds == 0) return;

    using value_type = typename FAB::value_type;

#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
    for (int j = 0; j < N_snds; ++j)
    {
        if (send_size[j] > 0)
        {
            char* dptr = send_data[j];
            auto const& cctc = *send_cctc[j];
            for (auto const& tag : cctc)
            {
                const amrex::Box& bx = tag.sbox;
                auto const sfab = src.array(tag.srcIndex);
                auto pfab = amrex::makeArray4((value_type*)(dptr),bx,ncomp);
                amrex::LoopConcurrentOnCpu( bx, ncomp,
                [=] (int ii, int jj, int kk, int n) noexcept
                {
                    pfab(ii,jj,kk,n) = sfab(ii,jj,kk,n+scomp);
                });
                dptr += (bx.numPts() * ncomp * sizeof(value_type));
            }
            BL_ASSERT(dptr <= send_data[j] + send_size[j]);
        }
    }
}

template <class FAB>
void
unpack_recv_buffer_cpu (amrex::FabArray<FAB>& dst, int dcomp, int ncomp,
                        std::vector<char*> const& recv_data,
                        std::vector<std::size_t> const& recv_size,
                        std::vector<CopyComTagsContainer const*> const& recv_cctc,
                        CpOp op, bool is_thread_safe)
{
    amrex::ignore_unused(recv_size);

    const int N_rcvs = recv_cctc.size();
    if (N_rcvs == 0) return;

    using value_type = typename FAB::value_type;

    if (is_thread_safe)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int k = 0; k < N_rcvs; ++k)
        {
            if (recv_size[k] > 0)
            {
                const char* dptr = recv_data[k];
                auto const& cctc = *recv_cctc[k];
                for (auto const& tag : cctc)
                {
                    const amrex::Box& bx  = tag.dbox;
                    FAB& dfab = dst[tag.dstIndex];
                    if (op == COPY)
                    {
                        dfab.template copyFromMem<amrex::RunOn::Host>(bx, dcomp, ncomp, dptr);
                    }
                    else
                    {
                        dfab.template addFromMem<amrex::RunOn::Host>(tag.dbox, dcomp, ncomp, dptr);
                    }
                    dptr += bx.numPts() * ncomp * sizeof(value_type);
                }
                BL_ASSERT(dptr <= recv_data[k] + recv_size[k]);
            }
        }
    }
    else
    {
        amrex::LayoutData<std::vector<VoidCopyTag> > recv_copy_tags;
        recv_copy_tags.define(dst.boxArray(),dst.DistributionMap());
        for (int k = 0; k < N_rcvs; ++k)
        {
            if (recv_size[k] > 0)
            {
                const char* dptr = recv_data[k];
                auto const& cctc = *recv_cctc[k];
                for (auto const& tag : cctc)
                {
                    recv_copy_tags[tag.dstIndex].push_back({dptr,tag.dbox});
                    dptr += tag.dbox.numPts() * ncomp * sizeof(value_type);
                }
                BL_ASSERT(dptr <= recv_data[k] + recv_size[k]);
            }
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
        {
            const auto& tags = recv_copy_tags[mfi];
            auto dfab = dst.array(mfi);
            for (auto const & tag : tags)
            {
                auto pfab = amrex::makeArray4((value_type*)(tag.p), tag.dbox, ncomp);
                if (op == COPY)
                {
                    amrex::LoopConcurrentOnCpu(tag.dbox, ncomp,
                    [=] (int i, int j, int k, int n) noexcept
                    {
                        dfab(i,j,k,n+dcomp) = pfab(i,j,k,n);
                    });
                }
                else
                {
                    amrex::LoopConcurrentOnCpu(tag.dbox, ncomp,
                    [=] (int i, int j, int k, int n) noexcept
                    {
                        dfab(i,j,k,n+dcomp) += pfab(i,j,k,n);
                    });
                }
            }
        }
    }
}


#endif
