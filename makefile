default: all

##################    VVVV    Change these    VVVV    ##################

# HLIBPRO_DIR := /home/nick/hlibpro-2.8.1
HLIBPRO_DIR := /home/nick/hlibpro-2.9
# HLIBPRO_DIR := /home/nick/hlibpro-2.9.1
# HLIBPRO_DIR := /home/nick/hlibpro-3.0
EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics3/include/eigen3
THREADPOOL_INCLUDE := /home/nick/repos/thread-pool # https://github.com/bshoshany/thread-pool
NALGER_HELPER_INCLUDE := /home/nick/repos/nalger_helper_functions/include

########################################################################

HLIBPRO_LIB := $(HLIBPRO_DIR)/lib
HLIBPRO_INCLUDE := $(HLIBPRO_DIR)/include
HLIBPRO_FLAGS := $(shell $(HLIBPRO_DIR)/bin/hlib-config --cflags --lflags)

PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

SRC_DIR  := ./src
OBJ_DIR  := ./obj
# OBJ_DIR  := .
# BUILD_DIR  := ./build
BUILD_DIR  := ./hlibpro_python_wrapper

LDFLAGS  = -shared -L$(HLIBPRO_LIB)
CXXFLAGS = -O3 -Wall -shared -fPIC -std=c++17 # <--- GOOD
# CXXFLAGS = -O3 -Wall -shared -fPIC -std=c++14
# CXXFLAGS = -O3 -Wall -shared -fPIC
# CXXFLAGS = -Wall -shared -fPIC -std=c++17 -g -fsanitize=address
# CXXFLAGS = -Wall -shared -fPIC -std=c++17 -g

LIBS := -lhpro -Wl,-rpath,$(HLIBPRO_LIB)

ALL_COMPILE_STUFF = $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
					-I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) -I$(NALGER_HELPER_INCLUDE) -I$(THREADPOOL_INCLUDE) \
					$(LDFLAGS) $(LIBS)

# HLIBPRO_BINDINGS_TARGET = $(addsuffix $(PYSUFFIX), hlibpro_bindings)
HLIBPRO_BINDINGS_TARGET = hlibpro_bindings.so


all: $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET): $(OBJ_DIR)/hlibpro_bindings.o $(OBJ_DIR)/grid_interpolate.o $(OBJ_DIR)/product_convolution_hmatrix.o # $(OBJ_DIR)/rbf_interpolation.o
	@echo 'Building target: $@'
	g++ -o "$@" $^ $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/hlibpro_bindings.o: $(SRC_DIR)/hlibpro_bindings.cpp $(SRC_DIR)/product_convolution_kernel.h $(SRC_DIR)/misc.h $(SRC_DIR)/rbf_interpolation.h
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
# 	cc -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/product_convolution_hmatrix.o: $(SRC_DIR)/product_convolution_hmatrix.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/grid_interpolate.o: $(SRC_DIR)/grid_interpolate.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
# 	cc -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

# $(OBJ_DIR)/rbf_interpolation.o: $(SRC_DIR)/rbf_interpolation.cpp
# 	@echo 'Building target: $@'
# 	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
# # 	cc -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
# 	@echo 'Finished building target: $@'
# 	@echo ' '


clean:
	-rm -rf $(OBJ_DIR)/product_convolution_hmatrix.o
	-rm -rf $(OBJ_DIR)/hlibpro_bindings.o
	-rm -rf $(OBJ_DIR)/grid_interpolate.o
# 	-rm -rf $(OBJ_DIR)/rbf_interpolation.o
	-rm -rf $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	-@echo ' '

.PHONY: all clean dependents
