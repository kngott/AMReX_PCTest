#ifndef AMINI_DEFS_H
#define AMINI_DEFS_H

#include <AMReX_FabArray.H>

enum CpOp { COPY = 0, ADD = 1 };
enum struct RunOn { Gpu, Cpu, Device=Gpu, Host=Cpu };

struct Dim3 { int x; int y; int z; };

template <class FAB>
struct FabCopyTag {
    FAB const* sfab;
    amrex::Box dbox;
    amrex::IntVect offset; // sbox.smallEnd() - dbox.smallEnd()
};

template <class T>
struct Array4CopyTag {
    amrex::Array4<T      > dfab;
    amrex::Array4<T const> sfab;
    amrex::Box dbox;
    amrex::Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4MaskCopyTag {
    amrex::Array4<T      > dfab;
    amrex::Array4<T const> sfab;
    amrex::Array4<int    > mask;
    amrex::Box dbox;
    amrex::Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Box const& box () const noexcept { return dbox; }
};

struct VoidCopyTag {
    char const* p;
    amrex::Box dbox;
};

template <class T>
struct Array4BoxTag {
    amrex::Array4<T> dfab;
    amrex::Box       dbox;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Box const& box () const noexcept { return dbox; }
};

//! Used by a bunch of routines when communicating via MPI.
struct CopyComTag
{
    amrex::Box dbox;
    amrex::Box sbox;
    int dstIndex;
    int srcIndex;
    CopyComTag () noexcept {}
    CopyComTag (const amrex::Box& db, const amrex::Box& sb, int didx, int sidx) noexcept
        : dbox(db), sbox(sb), dstIndex(didx), srcIndex(sidx) {}
    bool operator< (const CopyComTag& rhs) const noexcept {
        return (srcIndex < rhs.srcIndex) || ((srcIndex == rhs.srcIndex) && (
               (sbox.smallEnd() < rhs.sbox.smallEnd()
                           || ((sbox.smallEnd() == rhs.sbox.smallEnd()) && (
               (dstIndex < rhs.dstIndex) || ((dstIndex == rhs.dstIndex) && (
               (dbox.smallEnd() < rhs.dbox.smallEnd()))))))));
    }
};

typedef std::vector<CopyComTag> CopyComTagsContainer;
typedef std::map<int,CopyComTagsContainer> MapOfCopyComTagContainers;


#endif
