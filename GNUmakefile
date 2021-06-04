# AMREX_HOME defines the directory in which we will find all the AMReX code.
# If you set AMREX_HOME as an environment variable, this line will be ignored
AMREX_HOME ?= ../amrex
TOP = .

#DEBUG        = TRUE 
USE_OMP      = FALSE
TINY_PROFILE = TRUE
USE_MPI      = TRUE
USE_CUDA     = FALSE
COMP         = gcc
DIM          = 3

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(TOP)/PC/Make.package
INCLUDE_LOCATIONS += $(TOP)/PC
VPATH_LOCATIONS   += $(TOP)/PC

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
