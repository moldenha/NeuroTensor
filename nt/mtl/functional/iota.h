#ifdef NT_MTL_SUPPORTED
#ifndef NT_MTL_FUNCTIONAL_IOTA_H__
#define NT_MTL_FUNCTIONAL_IOTA_H__

#include "../utils.h"
#include "../../memory/device.h"

namespace nt::mtl{

void iota_int64(intrusive_ptr<DeviceMTLPrivate> dev, int64_t size, int64_t start, int64_t offset = 0);
void iota_int64(intrusive_ptr<DeviceMTLShared> dev, int64_t size, int64_t start, int64_t offset = 0);


}

#endif
#endif
