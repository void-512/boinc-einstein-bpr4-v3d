#include "brp/vulkan.hpp"
#include "brp/lut.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>
namespace brp {
static void check(VkResult result,const char* operation) {
    if(result!=VK_SUCCESS) throw std::runtime_error(std::string(operation)+": Vulkan error "+std::to_string(result));
}
static double cpu_time() { timespec t{}; clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
struct GPU::Impl {
    VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{};
    VkPhysicalDeviceProperties properties{}; VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    uint32_t family{}, timestamp_bits{};
    struct Buffer { VkBuffer buffer{}; VkDeviceMemory memory{}; void* mapped{}; VkDeviceSize allocation{}, size{}; };
    std::array<Buffer,8> buffers{};
    VkDescriptorSetLayout descriptors{}; VkPipelineLayout layout{}; VkDescriptorPool descriptor_pool{};
    VkDescriptorSet descriptor_set{}; VkCommandPool command_pool{}; VkCommandBuffer command{};
    VkFence fence{}; VkQueryPool queries{};
    std::array<VkPipeline,9> pipelines{};
    std::vector<uint64_t> timestamps;
    Params p{}; uint32_t wg{},tile{},batch{};bool stockham{},radix4{},output_scratch{};
    ~Impl() {
        if(device) {
            vkDeviceWaitIdle(device);
            for(auto pipeline:pipelines) if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
            if(fence) vkDestroyFence(device,fence,nullptr);
            if(queries) vkDestroyQueryPool(device,queries,nullptr);
            if(command_pool) vkDestroyCommandPool(device,command_pool,nullptr);
            if(descriptor_pool) vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
            if(layout) vkDestroyPipelineLayout(device,layout,nullptr);
            if(descriptors) vkDestroyDescriptorSetLayout(device,descriptors,nullptr);
            for(auto& b:buffers) {
                if(b.mapped) vkUnmapMemory(device,b.memory);
                if(b.buffer) vkDestroyBuffer(device,b.buffer,nullptr);
                if(b.memory) vkFreeMemory(device,b.memory,nullptr);
            }
            vkDestroyDevice(device,nullptr);
        }
        if(instance) vkDestroyInstance(instance,nullptr);
    }
    void buffer(uint32_t index,VkDeviceSize size) {
        auto& b=buffers[index]; b.size=size;
        if(size>properties.limits.maxStorageBufferRange) throw std::runtime_error("storage buffer exceeds device limit");
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=size;
        ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device,&ci,nullptr,&b.buffer),"create buffer");
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device,b.buffer,&req);
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<memory_properties.memoryTypeCount;++i) {
            auto flags=memory_properties.memoryTypes[i].propertyFlags;
            if((req.memoryTypeBits&(1u<<i)) && (flags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {type=i;break;}
        }
        if(type==UINT32_MAX) throw std::runtime_error("research backend requires coherent mapped memory");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        check(vkAllocateMemory(device,&ai,nullptr,&b.memory),"allocate memory"); b.allocation=req.size;
        check(vkBindBufferMemory(device,b.buffer,b.memory,0),"bind memory");
        check(vkMapMemory(device,b.memory,0,req.size,0,&b.mapped),"map memory");
    }
    void pipeline(uint32_t index,const char* name,uint32_t local) {
        std::ifstream file(std::string(BRP_SHADER_DIR)+"/"+name+".spv",std::ios::binary|std::ios::ate);
        if(!file || file.tellg()<=0 || size_t(file.tellg())%4) throw std::runtime_error("invalid/missing shader");
        std::vector<uint32_t> code(size_t(file.tellg())/4); file.seekg(0); file.read(reinterpret_cast<char*>(code.data()),code.size()*4);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=code.size()*4;mi.pCode=code.data();
        VkShaderModule module{};check(vkCreateShaderModule(device,&mi,nullptr,&module),"create shader");
        uint32_t values[3]={local,tile,stockham?1u:0u}; VkSpecializationMapEntry entries[3]={{0,0,4},{1,4,4},{2,8,4}};
        VkSpecializationInfo spec{};spec.mapEntryCount=index==8?3:(index==4?2:1);spec.pMapEntries=entries;spec.dataSize=sizeof(values);spec.pData=values;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=layout;
        ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module=module;ci.stage.pName="main";ci.stage.pSpecializationInfo=&spec;
        auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipelines[index]);
        vkDestroyShaderModule(device,module,nullptr);check(result,"create pipeline");
    }
    void barrier(VkPipelineStageFlags src=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VkPipelineStageFlags dst=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VkAccessFlags access=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask=((src&VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)?VK_ACCESS_SHADER_WRITE_BIT:0)|
                        ((src&VK_PIPELINE_STAGE_TRANSFER_BIT)?VK_ACCESS_TRANSFER_WRITE_BIT:0);
        b.dstAccessMask=access;
        vkCmdPipelineBarrier(command,src,dst,0,1,&b,0,nullptr,0,nullptr);
    }
    void dispatch(uint32_t pipeline_index,uint32_t groups,uint32_t y=1) {
        if(groups>properties.limits.maxComputeWorkGroupCount[0] || y>properties.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("dispatch exceeds device limits");
        vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[pipeline_index]);
        vkCmdPushConstants(command,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
        vkCmdDispatch(command,groups,y,1);
    }
    void init(const Params& params,std::span<const float> input,uint32_t local,uint32_t fft_tile,uint32_t replay,bool software,bool autosort,bool paired) {
        p=params;wg=local;tile=fft_tile;batch=replay;stockham=autosort;radix4=paired;
        if(stockham && tile) throw std::runtime_error("Stockham requires --tile 0");
#if BRP_SCATTER_PREPARE
        if(stockham) throw std::runtime_error("legacy scattered preparation does not support Stockham");
#endif
        if(!wg||wg>256 || !batch || batch>64 || (tile && (!std::has_single_bit(tile)||tile<64||tile>4096||tile>p.n)))
            throw std::runtime_error("invalid workgroup, FFT tile, or batch");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="BRP4 V3D research";app.apiVersion=VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;
        check(vkCreateInstance(&ci,nullptr,&instance),"create instance");
        uint32_t count{};check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"count devices");
        std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"enumerate devices");
        for(auto candidate:devices) { VkPhysicalDeviceProperties prop{};vkGetPhysicalDeviceProperties(candidate,&prop);
            if(std::string(prop.deviceName).find("V3D")!=std::string::npos) {physical=candidate;break;}}
        if(!physical && software && !devices.empty()) physical=devices[0];
        if(!physical) throw std::runtime_error("V3D device not found; GPU access must be outside sandbox");
        VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};prop2.pNext=&subgroup;
        vkGetPhysicalDeviceProperties2(physical,&prop2);properties=prop2.properties;
        vkGetPhysicalDeviceFeatures(physical,&features);vkGetPhysicalDeviceMemoryProperties(physical,&memory_properties);
        VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};supported.pNext=&supported13;
        vkGetPhysicalDeviceFeatures2(physical,&supported);
        if(!supported13.maintenance4) throw std::runtime_error("LocalSizeId shaders require maintenance4");
        if(properties.apiVersion<VK_API_VERSION_1_3 || wg>properties.limits.maxComputeWorkGroupInvocations ||
           wg>properties.limits.maxComputeWorkGroupSize[0] || sizeof(p)>properties.limits.maxPushConstantsSize ||
           tile*8>properties.limits.maxComputeSharedMemorySize || properties.limits.maxPerStageDescriptorStorageBuffers<8 ||
           properties.limits.maxDescriptorSetStorageBuffers<8) throw std::runtime_error("unsupported compute limits");
        uint32_t qc{};vkGetPhysicalDeviceQueueFamilyProperties(physical,&qc,nullptr);std::vector<VkQueueFamilyProperties> qprops(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&qc,qprops.data());family=UINT32_MAX;
        for(uint32_t i=0;i<qc;++i) if(qprops[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {family=i;timestamp_bits=qprops[i].timestampValidBits;break;}
        if(family==UINT32_MAX||!timestamp_bits) throw std::runtime_error("compute/timestamp queue missing");
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
        VkPhysicalDeviceVulkan13Features enabled13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};enabled13.maintenance4=VK_TRUE;di.pNext=&enabled13;
        check(vkCreateDevice(physical,&di,nullptr,&device),"create device");vkGetDeviceQueue(device,family,0,&queue);
        buffer(0,(p.unpadded+130)*4ull);buffer(1,p.n*4ull);buffer(2,p.n*8ull);
        buffer(3,BRP_STRIDED_TWIDDLES?p.n*4ull:(p.n-1)*8ull);
        buffer(4,p.n*2ull+p.fundamental*20ull);buffer(5,16+p.capacity*16ull);
        buffer(6,((p.steps+wg-1)/wg)*4ull);buffer(7,stockham?p.n*8ull:8ull);
        if(input.size()!=p.unpadded) throw std::runtime_error("input length mismatch");
        std::memcpy(buffers[0].mapped,input.data(),input.size_bytes());
        auto tw=static_cast<std::complex<float>*>(buffers[3].mapped);
        if(BRP_STRIDED_TWIDDLES) {
            for(uint32_t i=0;i<p.n/2;++i) {double angle=-2*std::numbers::pi*i/p.n;tw[i]={float(std::cos(angle)),float(std::sin(angle))};}
        } else {
            for(uint32_t width=2;width<=p.n;width*=2) for(uint32_t j=0;j<width/2;++j) {
                double angle=-2*std::numbers::pi*j/width;tw[width/2-1+j]={float(std::cos(angle)),float(std::sin(angle))};
            }
        }
        auto lut=reinterpret_cast<std::complex<float>*>(static_cast<float*>(buffers[0].mapped)+p.unpadded);
        for(uint32_t i=0;i<65;++i) lut[i]={sinSamples[i],cosSamples[i]};
        std::array<VkDescriptorSetLayoutBinding,8> bindings{};
        for(uint32_t i=0;i<8;++i) bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=bindings.size();li.pBindings=bindings.data();
        check(vkCreateDescriptorSetLayout(device,&li,nullptr,&descriptors),"descriptor layout");
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&descriptors;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&range;
        check(vkCreatePipelineLayout(device,&pli,nullptr,&layout),"pipeline layout");
        VkDescriptorPoolSize poolsize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,8};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&poolsize;
        check(vkCreateDescriptorPool(device,&dpi,nullptr,&descriptor_pool),"descriptor pool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=descriptor_pool;dai.descriptorSetCount=1;dai.pSetLayouts=&descriptors;
        check(vkAllocateDescriptorSets(device,&dai,&descriptor_set),"descriptor set");
        std::array<VkDescriptorBufferInfo,8> info{};std::array<VkWriteDescriptorSet,8> writes{};
        for(uint32_t i=0;i<8;++i) {info[i]={buffers[i].buffer,0,buffers[i].size};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet=descriptor_set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&info[i];}
        vkUpdateDescriptorSets(device,writes.size(),writes.data(),0,nullptr);
        const char* names[]={"resample","mean","prepare","fft","fft_local","power","harmonic","fft_stockham","fft_radix4"};
        for(uint32_t i=0;i<9;++i) if((i!=4||tile)&&(i!=7||stockham)&&(i!=8||radix4)) pipeline(i,names[i],wg);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;
        check(vkCreateCommandPool(device,&cpi,nullptr,&command_pool),"command pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=command_pool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&cai,&command),"command buffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fi,nullptr,&fence),"fence");
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qpi.queryType=VK_QUERY_TYPE_TIMESTAMP;qpi.queryCount=5*batch;
        check(vkCreateQueryPool(device,&qpi,nullptr,&queries),"timestamp pool");
        timestamps.resize(5*batch);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(command,&begin),"begin command");
        vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&descriptor_set,0,nullptr);
        vkCmdResetQueryPool(command,queries,0,5*batch);
        for(uint32_t r=0;r<batch;++r) {
            if(r) barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdFillBuffer(command,buffers[5].buffer,0,8,0);
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,r*5);
            p.stage=stockham?0x80000000u:0;
            dispatch(0,(p.steps+wg-1)/wg);barrier();dispatch(1,1);barrier();dispatch(2,(p.n+wg-1)/wg);barrier();
            vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,r*5+1);
            if(tile) {dispatch(4,p.n/tile);barrier();}
            uint32_t alternate=0;
            for(uint32_t width=tile?tile*2:2;width<=p.n;) {
                bool paired_stage=radix4&&width<=p.n/2;
                p.stage=(paired_stage?width*2:width)|(stockham?alternate:0);
                uint32_t invocations=paired_stage?p.n/4:p.n/2;
                dispatch(paired_stage?8:(stockham?7:3),(invocations+wg-1)/wg);barrier();
                alternate^=0x80000000u;width*=paired_stage?4:2;
            }
            output_scratch=stockham&&alternate;
            p.stage=output_scratch?0x80000000u:0;
            vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,r*5+2);
            dispatch(5,(p.n/2+wg-1)/wg);barrier();
            vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,r*5+3);
            dispatch(6,(p.fundamental+wg-1)/wg,5);
            vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,r*5+4);
            // Later replay reads/writes the same resources. Order both WAR and RAW.
            barrier();
        }
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);
        check(vkEndCommandBuffer(command),"end command");
    }
};
GPU::GPU(const Params& p,std::span<const float> input,uint32_t wg,uint32_t tile,uint32_t batch,bool software,bool stockham,bool radix4):impl(std::make_unique<Impl>()) {
    impl->init(p,input,wg,tile,batch,software,stockham,radix4);
}
GPU::~GPU()=default;
Timings GPU::run() {
    auto& g=*impl;
    auto start=std::chrono::steady_clock::now();double cpu_start=cpu_time();
    check(vkResetFences(g.device,1,&g.fence),"reset fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&g.command;
    auto submit_start=std::chrono::steady_clock::now();
    check(vkQueueSubmit(g.queue,1,&submit,g.fence),"submit");
    double submit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-submit_start).count();
    check(vkWaitForFences(g.device,1,&g.fence,VK_TRUE,UINT64_MAX),"wait for GPU");
    Timings t{};t.submit_ms=submit_ms;
    auto& timestamps=g.timestamps;
    check(vkGetQueryPoolResults(g.device,g.queries,0,timestamps.size(),timestamps.size()*8,timestamps.data(),8,VK_QUERY_RESULT_64_BIT),"read timestamps");
    uint64_t mask=g.timestamp_bits==64?UINT64_MAX:((1ull<<g.timestamp_bits)-1);
    for(uint32_t r=0;r<g.batch;++r) for(uint32_t s=0;s<4;++s) t.stage_ms[s]+=double((timestamps[r*5+s+1]-timestamps[r*5+s])&mask)*g.properties.limits.timestampPeriod/1e6;
    t.cpu_ms=cpu_time()-cpu_start;t.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return t;
}
Snapshot GPU::snapshot() const {
    auto& g=*impl;Snapshot s;
    auto floats=[&](uint32_t b,size_t count){auto ptr=static_cast<float*>(g.buffers[b].mapped);return std::vector<float>(ptr,ptr+count);};
    s.resampled=floats(1,g.p.n);s.power=floats(4,g.p.n/2);
    auto summed=static_cast<float*>(g.buffers[4].mapped)+g.p.n/2;s.summed.assign(summed,summed+5*g.p.fundamental);
    auto fft=static_cast<std::complex<float>*>(g.buffers[g.output_scratch?7:2].mapped);s.fft.assign(fft,fft+g.p.n/2+1);
    auto header=static_cast<uint32_t*>(g.buffers[5].mapped);
    if(header[1] || header[0]>g.p.capacity) throw std::runtime_error("GPU invalid resample index or candidate overflow; no valid result");
    // std430 aligns the uvec4 runtime array to 16 bytes after the two scalars.
    auto c=reinterpret_cast<Candidate*>(header+4);s.candidates.assign(c,c+header[0]);
    std::sort(s.candidates.begin(),s.candidates.end(),[](auto a,auto b){return a.level!=b.level?a.level<b.level:a.bin<b.bin;});return s;
}
std::string GPU::device_info() const {
    auto& g=*impl;std::ostringstream out;
    out<<g.properties.deviceName<<"; Vulkan "<<VK_API_VERSION_MAJOR(g.properties.apiVersion)<<'.'<<VK_API_VERSION_MINOR(g.properties.apiVersion)
       <<"; subgroup="<<g.subgroup.subgroupSize<<"; FP64="<<g.features.shaderFloat64<<"; int64="<<g.features.shaderInt64
       <<"; shared="<<g.properties.limits.maxComputeSharedMemorySize<<"; timestamp_ns="<<g.properties.limits.timestampPeriod;
    return out.str();
}
uint64_t GPU::allocated_bytes() const {uint64_t size=0;for(auto& b:impl->buffers) size+=b.allocation;return size;}
}
