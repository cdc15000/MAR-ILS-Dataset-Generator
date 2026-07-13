# MAR ILS Lab Harness — C/C++
# ASTM WKXXXXX Rev 05
#
# Prerequisites: libhdf5 (https://www.hdfgroup.org/downloads/hdf5/)
#
# Usage:
#   make                    # build with the example FBP implementation
#   make MAR_SRC=my_mar.c   # build with your MAR implementation
#   make clean

CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra
LDFLAGS  ?=

# Lab's MAR source file — override to plug in your implementation
MAR_SRC  ?= apply_mar_example.c

# HDF5 — adjust paths for your installation
# macOS (Homebrew):  brew install hdf5
# Linux (apt):       sudo apt install libhdf5-dev
HDF5_CFLAGS  ?= $(shell pkg-config --cflags hdf5 2>/dev/null || echo -I/usr/include/hdf5/serial)
HDF5_LDFLAGS ?= $(shell pkg-config --libs   hdf5 2>/dev/null || echo -lhdf5)

TARGET = mar_harness

SRCS = mar_harness.c $(MAR_SRC)
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(HDF5_LDFLAGS) -lm

%.o: %.c apply_mar.h
	$(CC) $(CFLAGS) $(HDF5_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
