MODULE_TOPDIR = ../..

PGM = i.sar.coregistration

include $(MODULE_TOPDIR)/include/Make/Script.make

# Compute library loaded by the script through ctypes, installed in
# $(ETC)/$(PGM) (copied by "make install" with the rest of ETCDIR).
SARLIB = $(ETCDIR)/libsarcoreg$(SHLIB_SUFFIX)

default: script $(SARLIB)

$(OBJDIR)/sarcoreg_cl.h: sarcoreg_core.h sarcoreg_kernels.cl | $(OBJDIR)
	cat sarcoreg_core.h sarcoreg_kernels.cl | \
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' > $@

$(SARLIB): sarcoreg.c sarcoreg.h sarcoreg_core.h $(OBJDIR)/sarcoreg_cl.h | $(ETCDIR)
	$(CC) -O3 -std=gnu11 $(SHLIB_CFLAGS) $(OPENMP_CFLAGS) $(OCLINCPATH) \
		-I$(OBJDIR) -I. -o $@ sarcoreg.c $(SHLIB_LD_FLAGS) -shared \
		$(OCLLIBPATH) $(OCLLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB) $(MATHLIB)

