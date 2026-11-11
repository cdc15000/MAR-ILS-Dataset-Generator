# MAR ILS laboratory harness, C (ASTM WK101378 draft Rev 06, v9 dataset)
#
# Prerequisite: libhdf5 (macOS: brew install hdf5; Debian/Ubuntu: apt install libhdf5-dev)
#
#   make                     build with the example (reference reconstruction, no MAR)
#   make MAR_SRC=my_mar.c    build with your implementation
#   make OPENMP=1            parallelize the reference backprojection (needs an OpenMP compiler)
#   make clean

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
MAR_SRC ?= apply_mar_example.c

HDF5_CFLAGS  ?= $(shell pkg-config --cflags hdf5 2>/dev/null || h5cc -showconfig 2>/dev/null | sed -n 's/^ *Installation point: *\(.*\)/-I\1\/include/p' || echo -I/usr/include/hdf5/serial)
HDF5_LDFLAGS ?= $(shell pkg-config --libs hdf5 2>/dev/null || h5cc -showconfig 2>/dev/null | sed -n 's/^ *Installation point: *\(.*\)/-L\1\/lib -lhdf5/p' || echo -lhdf5)

ifdef OPENMP
CFLAGS += -fopenmp
LDLIBS += -fopenmp
endif

TARGET = mar_harness
SRCS   = mar_harness.c reference_fbp.c $(MAR_SRC)
OBJS   = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(HDF5_LDFLAGS) $(LDLIBS) -lm

%.o: %.c apply_mar.h reference_fbp.h
	$(CC) $(CFLAGS) $(HDF5_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
