#ifndef _NT_LAYERS_H_
#define _NT_LAYERS_H_

#include "layers/Linear.h"
#include "layers/Identity.h"
#include "layers/Conv1D.h"
#include "layers/Conv2D.h"
#include "layers/Conv3D.h"
#include "layers/ConvTranspose1D.h"
#include "layers/ConvTranspose2D.h"
#include "layers/ConvTranspose3D.h"
#include "layers/Unfold1D.h"
#include "layers/Unfold2D.h"
#include "layers/Unfold3D.h"
#include "layers/Fold.h"
#include "layers/Softplus.h"
#include "layers/Dropout.h"
#include "layers/BatchNorm1D.h"
#include "layers/Functional.h"

#include "layer_reflect/layer_registry.hpp"
#include "layer_reflect/reflect_macros.h"

#include "functional.h"

#endif
