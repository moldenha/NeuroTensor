CC = g++
STL_VER = -std=c++17
USE_ENCRYPTION := 0
USE_TBB_PARALLEL := 1

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
MAIN_FILE := $(mkfile_dir)main.cpp
OBJCS_DIR := $(mkfile_dir)objcs
MAIN_OBJ := $(OBJCS_DIR)/main.o
SRC_DIR := $(mkfile_dir)src/

ARRAYREF_FILES := $(SRC_DIR)refs/ArrayRef.cpp $(SRC_DIR)refs/ArrayRef.h
ARRAYREF_OBJCS :=  $(OBJCS_DIR)/refs/ArrayRef.o

SIZEREF_FILES := $(SRC_DIR)refs/SizeRef.cpp $(SRC_DIR)refs/SizeRef.h
SIZEREF_OBJCS :=  $(OBJCS_DIR)/refs/SizeRef.o


UTIL_FILES := $(SRC_DIR)utils/utils.cpp $(SRC_DIR)utils/utils.h 
UTIL_OBJCS := $(OBJCS_DIR)/utils/utils.o

TENSOR_FILES := $(SRC_DIR)Tensor.cpp $(SRC_DIR)Tensor.h
TENSOR_OBJCS := $(OBJCS_DIR)/Tensor.o

LOAD_FILES := $(SRC_DIR)functional/load.cpp $(SRC_DIR)functional/functional.h
LOAD_OBJCS := $(OBJCS_DIR)/functional/load.o

SAVE_FILES := $(SRC_DIR)functional/save.cpp $(SRC_DIR)functional/functional.h
SAVE_OBJCS := $(OBJCS_DIR)/functional/save.o

FUNCTIONAL_FILES := $(SRC_DIR)functional/functional.cpp $(SRC_DIR)functional/functional.h
FUNCTIONAL_OBJCS := $(OBJCS_DIR)/functional/functional.o

FUNCTIONAL_OPERATOR_FILES := $(SRC_DIR)functional/functional_operator.cpp $(SRC_DIR)functional/functional_operator.h
FUNCTIONAL_OPERATOR_OBJCS := $(OBJCS_DIR)/functional/functional_operator.o


# FUNCTIONAL_MATMULT_FILES := $(SRC_DIR)functional/functional_matmult.cpp $(SRC_DIR)functional/functional_matmult.h
# FUNCTIONAL_MATMULT_OBJCS := $(OBJCS_DIR)/functional/functional_matmult.o


WRBITS_FILES := $(SRC_DIR)functional/wrbits.cpp $(SRC_DIR)functional/wrbits.h
WRBITS_OBJCS := $(OBJCS_DIR)/functional/wrbits.o

DTYPE_FILES := $(SRC_DIR)dtype/DType.cpp $(SRC_DIR)dtype/DType.h
DTYPE_OBJCS := $(OBJCS_DIR)/dtype/DType.o

DTYPE_OPERATOR_FILES := $(SRC_DIR)dtype/DType_operators.h $(SRC_DIR)dtype/DType_operators.cpp
DTYPE_OPERATOR_OBJCS := $(OBJCS_DIR)/dtype/DType_operators.o

ARRAYVOID_FILES := $(SRC_DIR)dtype/ArrayVoid.cpp $(SRC_DIR)dtype/ArrayVoid.h
ARRAYVOID_OBJCS := $(OBJCS_DIR)/dtype/ArrayVoid.o

DTYPELIST_FILES := $(SRC_DIR)dtype/DType_list.cpp $(SRC_DIR)dtype/DType_list.h
DTYPELIST_OBJCS := $(OBJCS_DIR)/dtype/DType_list.o

SCALAR_FILES := $(SRC_DIR)dtype/Scalar.cpp $(SRC_DIR)dtype/Scalar.h
SCALAR_OBJCS := $(OBJCS_DIR)/dtype/Scalar.o

RANGES_FILES := $(SRC_DIR)dtype/ranges.cpp $(SRC_DIR)dtype/ranges.h
RANGES_OBJCS := $(OBJCS_DIR)/dtype/ranges.o

PERMUTE_FILES := $(SRC_DIR)permute/permute.h $(SRC_DIR)permute/permute.cpp
PERMUTE_OBJCS := $(OBJCS_DIR)/permute/permute.o

PERMUTE_OLD_FILES := $(SRC_DIR)permute/permute_old.h $(SRC_DIR)permute/permute_old.cpp
PERMUTE_OLD_OBJCS := $(OBJCS_DIR)/permute/permute_old.o


TYPES_FILES := $(SRC_DIR)types/Types.cpp $(SRC_DIR)types/Types.h
TYPES_OBJCS := $(OBJCS_DIR)/types/Types.o 

BF_FILES := $(SRC_DIR)functional/bf.cpp $(SRC_DIR)functional/bf.h 
BF_OBJCS := $(OBJCS_DIR)/functional/bf.o

CONVERT_FILES := $(SRC_DIR)convert/Convert.h $(SRC_DIR)convert/std_convert.h $(SRC_DIR)convert/convert.cpp
CONVERT_OBJCS := $(OBJCS_DIR)/convert/Convert.o

#these are the layer files:
GENERAL_LAYER_FILES := $(SRC_DIR)layers/layers.h $(SRC_DIR)layers/Layer.cpp
GENERAL_LAYER_OBJCS := $(OBJCS_DIR)/layers/Layer.o

LINEAR_LAYER_FILES := $(SRC_DIR)layers/layers.h $(SRC_DIR)layers/Linear.cpp
LINEAR_LAYER_OBJCS := $(OBJCS_DIR)/layers/Linear.o

ACTIVATION_FUNCTION_LAYER_FILES := $(SRC_DIR)layers/layers.h $(SRC_DIR)layers/ActivationFunction.cpp
ACTIVATION_FUNCTION_LAYER_OBJCS := $(OBJCS_DIR)/layers/ActivationFunction.o


FOLD_LAYER_FILES := $(SRC_DIR)layers/layers.h $(SRC_DIR)layers/Fold.cpp
FOLD_LAYER_OBJCS := $(OBJCS_DIR)/layers/Fold.o 

SOURCES = $(SRC_DIR)refs/ArrayRef.cpp
SOURCES += $(SRC_DIR)refs/SizeRef.cpp
SOURCES += $(SRC_DIR)utils/utils.cpp
SOURCES += $(SRC_DIR)Tensor.cpp
SOURCES += $(SRC_DIR)functional/functional.cpp
SOURCES += $(SRC_DIR)functional/functional_operator.cpp
# SOURCES += $(SRC_DIR)functional/functional_matmult.cpp
ifeq ($(USE_ENCRYPTION), 1)
	SOURCES += $(SRC_DIR)functional/bf.cpp
endif
SOURCES += $(SRC_DIR)functional/load.cpp
SOURCES += $(SRC_DIR)functional/save.cpp
SOURCES += $(SRC_DIR)functional/wrbits.cpp
SOURCES += $(SRC_DIR)dtype/ArrayVoid.cpp
SOURCES += $(SRC_DIR)dtype/DType.cpp
SOURCES += $(SRC_DIR)dtype/DType_operators.cpp
SOURCES += $(SRC_DIR)dtype/DType_list.cpp
SOURCES += $(SRC_DIR)dtype/ranges.cpp
SOURCES += $(SRC_DIR)dtype/Scalar.cpp
SOURCES += $(SRC_DIR)types/Types.cpp
SOURCES += $(SRC_DIR)permute/permute_old.cpp
SOURCES += $(SRC_DIR)permute/permute.cpp
SOURCES += $(SRC_DIR)convert/Convert.cpp
SOURCES += $(SRC_DIR)layers/Linear.cpp
SOURCES += $(SRC_DIR)layers/Layer.cpp
SOURCES += $(SRC_DIR)layers/ActivationFunction.cpp
SOURCES += $(SRC_DIR)layers/Fold.cpp

# SOURCES += main.cpp

OUT_DIRS = $(sort $(dir $(subst $(SRC_DIR), $(OBJCS_DIR)/, $(SOURCES))))

#OBJS = $(addprefix $(OBJCS_DIR)/,$(addsuffix .0,$(basename $(pathsubst %,%,$(SOURCES)))))
OBJCS = $(subst $(SRC_DIR), $(OBJCS_DIR)/, $(SOURCES:%.cpp=%.o))
OBJCS_TOTAL = $(basename $(subst $(SRC_DIR), $(OBJCS_DIR)/, $(SOURCES)))
SRCS_TOTAL = $(basename $(SOURCES))
#OBJCS_TOTAL = $(addprefix $(OBJCS_DIR)/,$(basename $(SOURCES)))

$(info objcs_total : $(OBJCS_TOTAL))

REG_COMPILE_CFLAGS = -c
ifeq ($(USE_ENCRYPTION), 1)
	REG_COMPILE_CFLAGS += -DUSE_ENCRYPTION_UINT8
endif

ifeq ($(USE_TBB_PARALLEL), 1)
	REG_COMPILE_CFLAGS += -DUSE_PARALLEL
endif

REG_COMPILE := $(CC) $(STL_VER) $(REG_COMPILE_CFLAGS)
# OBJCS := $(OBJCS_DIR)ArrayRef.o $(OBJCS_DIR)SizeRef.o $(OBJCS_DIR)Itterator.o $(OBJCS_DIR)Tensor.o
CC_FLAGS := -Wall $(STL_VER) 
ifeq ($(USE_ENCRYPTION), 1)
	CC_FLAGS += -DUSE_ENCRYPTION_UINT8
endif
FILE_DEPENDENCIES := $(MAIN_FILE) $(TENSOR_FILES) $(SIZEREF_FILES) $(ITTERATOR_FILES) $(ARRAYREF_FILES) $(GENERAL_LAYER_FILES) $(LINEAR_LAYER_FILES)
ifeq ($(USE_TBB_PARALLEL), 1)
	CC_FLAGS += -DUSE_PARALLEL
	CC_FLAGS += -ltbb
endif

build: $(OUT_DIRS) $(OBJCS) $(MAIN_OBJ)
	$(CC) $(CC_FLAGS) $(OBJCS) $(MAIN_OBJ) -o main
#	$(CC) $(CC_FLAGS) $(OBJCS) $(MAIN_OBJ) -fsanitize=address -o main

# output: $(OBJCS) $(FILE_DEPENDENCES)
# 	$(CC) $(CC_FLAGS) $(OBJCS) -o main

$(MAIN_OBJ): $(MAIN_FILE) test.h $(SOURCES)
	$(REG_COMPILE) $(MAIN_FILE) -o $(MAIN_OBJ)

# Now this rule also requres the object sub-dirs to be created first
# $(OBJCS): $(basename $(subst $(OBJCS_DIR)/, $(SRC_DIR), $@))
# 	$(REG_COMPILE) $(basename $(subst $(OBJCS_DIR)/, $(SRC_DIR), $@)).cpp -o $@

# $(OBJCS): %.cpp
# 	$(REG_COMPILE) $(basename $(subst $(OBJCS_DIR)/, $(SRC_DIR), $@)).cpp -o $@


$(ARRAYREF_OBJCS): $(ARRAYREF_FILES)
	$(REG_COMPILE) $(SRC_DIR)refs/ArrayRef.cpp -o $@

$(SIZEREF_OBJCS): $(SIZEREF_FILES)
	$(REG_COMPILE) $(SRC_DIR)refs/SizeRef.cpp -o $@

$(UTIL_OBJCS): $(UTIL_FILES)
	$(REG_COMPILE) $(SRC_DIR)utils/utils.cpp -o $@

$(TENSOR_OBJCS): $(TENSOR_FILES)
	$(REG_COMPILE) $(SRC_DIR)Tensor.cpp -o $@

$(LOAD_OBJCS): $(LOAD_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/load.cpp -o $@

$(SAVE_OBJCS): $(SAVE_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/save.cpp -o $@

$(FUNCTIONAL_OBJCS): $(FUNCTIONAL_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/functional.cpp -o $@

$(FUNCTIONAL_OPERATOR_OBJCS): $(FUNCTIONAL_OPERATOR_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/functional_operator.cpp -o $@

# $(FUNCTIONAL_MATMULT_OBJCS): $(FUNCTIONAL_MATMULT_FILES)
# 	$(REG_COMPILE) -mavx2 -mavx512f -mavx512dq $(SRC_DIR)functional/functional_matmult.cpp -o $@

$(WRBITS_OBJCS): $(WRBITS_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/wrbits.cpp -o $@

$(DTYPE_OBJCS): $(DTYPE_FILES)
	$(REG_COMPILE) $(SRC_DIR)dtype/DType.cpp -o $@

$(DTYPE_OPERATOR_OBJCS): $(SRC_DIR)dtype/DType_operators.cpp $(SRC_DIR)dtype/DType_operators.h 
	$(REG_COMPILE) $(SRC_DIR)dtype/DType_operators.cpp -o $@

$(ARRAYVOID_OBJCS): $(ARRAYVOID_FILES)
	$(REG_COMPILE) $(SRC_DIR)dtype/ArrayVoid.cpp -o $@

$(DTYPELIST_OBJCS): $(DTYPELIST_FILES)
	$(REG_COMPILE) $(SRC_DIR)dtype/DType_list.cpp -o $@

$(RANGES_OBJCS): $(RANGES_FILES)
	$(REG_COMPILE) $(SRC_DIR)dtype/ranges.cpp -o $@

$(SCALAR_OBJCS): $(SCALAR_FILES)
	$(REG_COMPILE) $(SRC_DIR)dtype/Scalar.cpp -o $@

$(PERMUTE_OLD_OBJCS): $(PERMUTE_OLD_FILES)
	$(REG_COMPILE) $(SRC_DIR)permute/permute_old.cpp -o $@
$(PERMUTE_OBJCS): $(PERMUTE_FILES)
	$(REG_COMPILE) $(SRC_DIR)permute/permute.cpp -o $@
$(TYPES_OBJCS): $(TYPES_FILES)
	$(REG_COMPILE) $(SRC_DIR)types/Types.cpp -o $@

$(BF_OBJCS): $(BF_FILES)
	$(REG_COMPILE) $(SRC_DIR)functional/bf.cpp -o $@

$(CONVERT_OBJCS): $(CONVERT_FILES)
	$(REG_COMPILE) $(SRC_DIR)convert/Convert.cpp -o $@

$(GENERAL_LAYER_OBJCS): $(GENERAL_LAYER_FILES)
	$(REG_COMPILE) $(SRC_DIR)layers/Layer.cpp -o $@


$(LINEAR_LAYER_OBJCS): $(LINEAR_LAYER_FILES)
	$(REG_COMPILE) $(SRC_DIR)layers/Linear.cpp -o $@

$(ACTIVATION_FUNCTION_LAYER_OBJCS): $(ACTIVATION_FUNCTION_LAYER_FILES)
	$(REG_COMPILE) $(SRC_DIR)layers/ActivationFunction.cpp -o $@

$(FOLD_LAYER_OBJCS): $(FOLD_LAYER_FILES)
	$(REG_COMPILE) $(SRC_DIR)layers/Fold.cpp -o $@

# Rule to create the object dir folders
$(OUT_DIRS):
	mkdir -p $@

clean:
	rm -rf $(OBJCS_DIR)
