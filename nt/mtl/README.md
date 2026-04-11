# NeuroTensor Metal Shader Library

The actual use of apple's metal-cpp to run a single function is rather long, can make bad cache allocation, and can make your code more error-prone (if used incorrectly).
For that reason, NeuroTensor has a specialized set of instructions and abstractions built on top of metal-cpp for using and running functions. The instructions on how to use them (and why they need to be used) are below.


## Metal-CPP installation

How to make sure it works out of the box:
- Ensure developer tools are installed: `sudo xcode-select --install`
- Make sure to install the `XCode` app
- Select XCode as the active developer directory: `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
- Verify with: `xcrun --find metal`
- Ensure you accept XCode license: `xcrun --find metal`
- You may also need to download the metal toolchain with this: `xcodebuild -downloadComponent MetalToolchain`
- Using an example:
    - Downlaod a git repository: git clone --recurse https://github.com/moldenha/mtl/ mtl
    - From there, cd to the the examples with the command `cd mtl/add3` and run:
```
cmake -S . -B build
cd build
cmake --build . --config Release
./[target]
```

**There is a repository at https://github.com/moldenha/mtl that shows how to use metal-cpp and install and run a test, the file at add3/CMakeLists.txt will show how to use a cmake file too**

## Running a kernel with bare metal-cpp

Below is an example and instruction set of how metal-cpp is normally used and can be used to run a function. It gives light to why an abstraction layer has been created within NeuroTensor for metal-cpp.

**Example Kernel name:** `add_num_kernel_float`

```c++
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstdint>

struct ThreadDispatchConfig {
    MTL::Size gridSize;
    MTL::Size threadgroupSize;
};

ThreadDispatchConfig computeThreadDispatchConfig(int64_t N, MTL::Device* device) {
    // Get max total threads allowed in a single threadgroup
    MTL::Size maxSize = device->maxThreadsPerThreadgroup();
    uint64_t maxThreadsPerGroup =
    int64_t(maxSize.width) * uint64_t(maxSize.height) * uint64_t(maxSize.depth);
    uint32_t max1DThreads = maxSize.width;


    if(N <= uint64_t(max1DThreads)) {
        // Pick a reasonable power-of-two size <= max
        uint32_t groupSize = 1;
        while (groupSize * 2 <= max1DThreads && groupSize * 2 <= N)
            groupSize *= 2;

        uint32_t numGroups = (N + groupSize - 1) / groupSize;

        return ThreadDispatchConfig {
            .gridSize = MTL::Size::Make(N, 1, 1),
            .threadgroupSize = MTL::Size::Make(groupSize, 1, 1)
        };
    }else {
        // split across 3D grid
        ThreadDispatchConfig config;
        uint64_t threadsRemaining = N;

        uint32_t gridX = std::min<uint64_t>(threadsRemaining, UINT32_MAX);
        threadsRemaining = (threadsRemaining + gridX - 1) / gridX;

        uint32_t gridY = std::min<uint64_t>(threadsRemaining, UINT32_MAX);
        threadsRemaining = (threadsRemaining + gridY - 1) / gridY;

        uint32_t gridZ = std::min<uint64_t>(threadsRemaining, UINT32_MAX);

        config.gridSize = MTL::Size::Make(gridX, gridY, gridZ);

        // Pick a threadgroup size in 3D (tgX * tgY * tgZ <= maxThreadsPerGroup)
        uint32_t tgX = 1, tgY = 1, tgZ = 1;

        // maximize tgX
        while (tgX * 2 <= gridX && tgX * 2 <= maxTG.width) tgX *= 2;
        // maximize tgY
        while (tgX * tgY * 2 <= maxThreadsPerGroup && tgY * 2 <= gridY && tgY * 2 <= maxTG.height) tgY *= 2;
        // maximize tgZ
        while (tgX * tgY * tgZ * 2 <= maxThreadsPerGroup && tgZ * 2 <= gridZ && tgZ * 2 <= maxTG.depth) tgZ *= 2;

        config.threadgroupSize = MTL::Size::Make(tgX, tgY, tgZ);
        return config;
    }

}

// an example of a function that add's 3 to each number
void add_num(float adding){
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::Buffer* staging_in = device->newBuffer(sizeof(float) * 10, MTL::ResourceStorageModeShared);
    MTL::Buffer* in_buffer = device->newBuffer(sizeof(float) * 10, MTL::ResourceStorageModePrivate);
    MTL::Buffer* out_buffer = device->newBuffer(sizeof(float) * 10, MTL::ResourceStorageModePrivate);
    MTL::Buffer* staging_out = device->newBuffer(sizeof(float) * 10, MTL::ResourceStorageModeShared);
    float* begin = staging_in->contents();
    for(int i = 0; i < 10; ++i){
        begin[i] = static_cast<float>(i+1);
    }


    NS::Error* error = nullptr;
    MTL::Library* library = nt::utils::mtl::nt_mtl_device->newLibrary(NS::String::string(METALLIB_PATH, NS::UTF8StringEncoding), nullptr);

    MTL::Function* kernelFunc = library->newFunction(NS::String::string("add_num_kernel_float", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipeline = nt::utils::mtl::nt_mtl_device->newComputePipelineState(kernelFunc, &error);

    MTL::CommandQueue* queue = nt::utils::mtl::nt_mtl_device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    
    // transfering shared to private
    MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
    blit->copyFromBuffer(staging_in, 0, in_buffer, 0, sieof(float) * 10);
    blit->endEncoding();

    //compute pass
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(in_buffer, 0, 0);
    encoder->setBuffer(out_buffer, 0, 1);
    encoder->setBytes(&adding, sizeof(float), 2);
    
    int N = 10; // 10 numbers
    ThreadDispatchConfig config = computeThreadDispatchConfig(N, device);
    encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // transfering shared to private
    MTL::BlitCommandEncoder* blit_out = commandBuffer->blitCommandEncoder();
    blit_out->copyFromBuffer(out_buffer, 0, staging_out, 0, sieof(float) * 10);
    blit_out->endEncoding();

    float* outData = static_cast<float*>(staging_out->contents());
    for (int i = 0; i < N; ++i)
        std::cout << "output[" << i << "] = " << outData[i] << std::endl;
    


    if (queue) queue->release();
    // These may already be autoreleased. Avoid releasing unless explicitly retained.
    if (pipeline) pipeline->release();
    if (kernelFunc) kernelFunc->release() ;
    if (library) library->release();

    // Buffers are usually safe to release if created with newBuffer
    if (in_buffer) in_buffer->release();
    if (out_buffer) out_buffer->release();
    if (staging_in) staging_in->release();
    if (staging_out) staging_out->release();

    // Only call release if pool was created using alloc/init
    if (pool) pool->drain();  // safer than release
    if (device) device->release();
}
```

## Using NeuroTensor Metal Abstraction

Here the same thing as the previous section is going to be ran as above
```

#include <nt/mtl/abstraction.h>
#include <nt/memory/device.h>

void add_num(float number){
    using namespace nt;
    DeviceMTLShared staging_in;
    DeviceMTLPrivate in_buffer;
    DeviceMTLPrivate out_buffer;
    DeviceMTLShared staging_out;
    staging_in.allocate_memory(DType::Float32, 10);
    in_buffer.allocate_memory(DType::Float32, 10);
    out_buffer.allocate_memory(DType::Float32, 10);
    staging_out.allocate_memory(DType::Float32, 10);
    
    float* begin = reinterpret_cast<float*>(staging_in.get_memory());
    for(int i = 0; i < 10; ++i)
        begin[i] = static_cast<float>(i+1);

    mtl::abs::MetalContext& ctx = mtl::abs::MetalContext.instance();
    intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline("add_num_kernel_float");
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
    int N = 10;
    mtl::mtl_shared_to_private(staging_in, commandBuffer_, in_buffer);
    mtl::abs::encodeCommand(
        mtl::abs::EncoderOptions{
            .commandBuffer = commandBuffer_,
            .pipeline = pipeline_,
            .size = N,
            .encode_grid = false
        },
        mtl::abs::EncoderNonOwning<intrusive_ptr<MetalBuffer>>{
                    .val = in_buffer.get_buffer(), .offset = 0
        },
        mtl::abs::EncoderNonOwning<intrusive_ptr<MetalBuffer>>{
                    .val = out_buffer..get_buffer(), .offset = 0
        },
        mtl::abs::EncoderCapture<int64_t>{number} // handles what should and should not be owned by neurotensor until complete
    );
    ctx.run_command(commandBuffer_);
    mtl::synchronize(); // the command can be run with async = false, or with this
    // another option was: ctx.run_command(commandBuffer_, /*async = */ false);
    // NeuroTensor assumes all functions can be run with async
    // If memory from buffers in use needs to be used or changed, it automatically handles waiting for functions to complete
    mtl::mtl_private_to_shared(out_buffer, commandBuffer_, staging_out);
    float* outData = reinterpret_cast<float*>(staging_out.get_memory());
    for (int i = 0; i < N; ++i)
        std::cout << "output[" << i << "] = " << outData[i] << std::endl;
    

 
}

```

- Important note is that NeuroTensor will automatically handle async running so that stalling on the CPU does not happen until absolutely needed
- For this reason the Encoder built into NeuroTensor is given what to hold onto and what not to in order to avoid use after frees and things like that
- All releases and allocations specific to metal-cpp are handled internally and automatically upon program exit and start (Except user specific like floats and other memory specific to user use)


## How to make kernels
