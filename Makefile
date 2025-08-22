# Simple Makefile (C11) for Linux and Termux aarch64

APP=mb_demo
BUILD_DIR=build
SRC=main.c
HDR=mb.h
BIN=$(BUILD_DIR)/$(APP)

CC ?= cc
CFLAGS ?= -std=c11 -O3 -pipe -DNDEBUG \
	-Wall -Wextra -Werror -pedantic -Wconversion -Wsign-conversion -Wshadow -Wformat=2 \
	-fno-omit-frame-pointer
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS ?=
LDLIBS ?=

.PHONY: all clean run

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) $(HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS) $(LDLIBS)

run: $(BIN)
	$(BIN)

clean:
	rm -rf $(BUILD_DIR)