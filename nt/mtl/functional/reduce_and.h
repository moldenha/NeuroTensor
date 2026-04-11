#ifdef NT_MTL_SUPPORTED
#ifndef NT_MTL_FUNCTIONAL_REDUCE_AND_H__
#define NT_MTL_FUNCTIONAL_REDUCE_AND_H__

#include "../utils.h"
#include "../../memory/device.h"

namespace nt::mtl{

bool reduce_and_uint32(intrusive_ptr<DeviceMTLPrivate> dev, int64_t size, int64_t start, int64_t offset = 0);
bool reduce_and_uint32(intrusive_ptr<DeviceMTLShared> dev, int64_t size, int64_t start, int64_t offset = 0);


}

#endif
#endif
