/**
 * IntegralImage 算子功能测试（aclnn 接口）
 *
 * 数据格式（同事确认版）：
 *   输入 image: (H,W) 或 (H,W,C)，uint8/float16/float32，ND（HWC 布局）
 *   输出 sat  : (H+1,W+1) 或 (C,H+1,W+1)，int32（整数输入）/ float32（浮点输入）
 *
 * 验证点：
 *   1. 物理零填充边 sat[0][*]=sat[*][0]=0
 *   2. CPU 参考逐元素比对（含 3D 逐通道独立）
 *   3. 窄 dtype（uint8/int16/float16）Cast 拓宽正确性
 *   4. 多核列分块（W=384 => 3 核）与块起始补偿
 */
#include <iostream>
#include <cmath>
#include <cstring>
#include <typeinfo>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_integral_image.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclInit failed: " << ret << std::endl; return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtSetDevice failed: " << ret << std::endl; return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtCreateStream failed: " << ret << std::endl; return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMalloc failed: " << ret << std::endl; return ret);
    if (!hostData.empty()) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMemcpy failed: " << ret << std::endl; return ret);
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// IEEE 754 float -> half（测试用简化实现）
uint16_t FloatToHalf(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    if (exp == 0xFF) {
        return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
    }
    int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (e <= 0) {
        return static_cast<uint16_t>(sign);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
}

// 通用 2D 用例：输入 InT、输出 OutT，期望值以 double 存储（int/float 统一比较）
template <typename InT, typename OutT>
int RunCaseT(aclrtStream stream, int64_t H, int64_t W, const std::vector<InT>& input, aclDataType inAclType,
             aclDataType outAclType, const std::vector<double>& expected)
{
    const int64_t OH = H + 1;
    const int64_t OW = W + 1;

    aclTensor* imageTensor = nullptr;
    aclTensor* satTensor = nullptr;
    void* imageDev = nullptr;
    void* satDev = nullptr;
    std::vector<int64_t> inShape = {H, W};
    std::vector<int64_t> outShape = {OH, OW};

    int ret = CreateAclTensor(input, inShape, &imageDev, inAclType, &imageTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "create image tensor failed" << std::endl; return ret);
    ret = CreateAclTensor(std::vector<OutT>(), outShape, &satDev, outAclType, &satTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "create sat tensor failed" << std::endl; return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnIntegralImageGetWorkspaceSize(imageTensor, satTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed: " << ret << std::endl; return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "workspace malloc failed: " << ret << std::endl; return ret);
    }
    ret = aclnnIntegralImage(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclnnIntegralImage failed: " << ret << std::endl; return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "sync stream failed: " << ret << std::endl; return ret);

    std::vector<OutT> actual(OH * OW, 0);
    ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(OutT), satDev, actual.size() * sizeof(OutT),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "copy sat back failed: " << ret << std::endl; return ret);

#ifdef DUMP_SAT
    std::cout << "dump sat (first 5 rows, first 8 cols):" << std::endl;
    for (int64_t rr = 0; rr < 5 && rr < OH; rr++) {
        for (int64_t cc = 0; cc < 8 && cc < OW; cc++) {
            std::cout << static_cast<double>(actual[rr * OW + cc]) << " ";
        }
        std::cout << std::endl;
    }
#endif

    bool ok = true;
    for (int64_t i = 0; i < OH * OW; i++) {
        double got = static_cast<double>(actual[i]);
        double want = expected[i];
        double tol = (std::fabs(want) > 1.0) ? std::fabs(want) * 1e-4 : 1e-3;
        if (std::fabs(got - want) > tol) {
            ok = false;
            std::cout << "sat mismatch at " << i / OW << "," << i % OW << ": got " << got << " expected " << want
                      << std::endl;
            break;
        }
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(imageTensor);
    aclDestroyTensor(satTensor);
    aclrtFree(imageDev);
    aclrtFree(satDev);

    std::cout << (ok ? "[PASS]" : "[FAIL]") << " IntegralImage " << typeid(InT).name() << " H=" << H << " W=" << W
              << std::endl;
    return ok ? 0 : 1;
}

// 通用 3D 用例：输入 (H,W,C)，输出 (H+1,W+1,C)（HWC 布局）
template <typename InT, typename OutT>
int RunCase3DT(aclrtStream stream, int64_t C, int64_t H, int64_t W, const std::vector<InT>& input,
               aclDataType inAclType, aclDataType outAclType, const std::vector<double>& expected)
{
    const int64_t OH = H + 1;
    const int64_t OW = W + 1;

    aclTensor* imageTensor = nullptr;
    aclTensor* satTensor = nullptr;
    void* imageDev = nullptr;
    void* satDev = nullptr;
    std::vector<int64_t> inShape = {H, W, C};
    std::vector<int64_t> outShape = {OH, OW, C};

    int ret = CreateAclTensor(input, inShape, &imageDev, inAclType, &imageTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "create image tensor failed" << std::endl; return ret);
    ret = CreateAclTensor(std::vector<OutT>(), outShape, &satDev, outAclType, &satTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "create sat tensor failed" << std::endl; return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnIntegralImageGetWorkspaceSize(imageTensor, satTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed: " << ret << std::endl; return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "workspace malloc failed: " << ret << std::endl; return ret);
    }
    ret = aclnnIntegralImage(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclnnIntegralImage failed: " << ret << std::endl; return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "sync stream failed: " << ret << std::endl; return ret);

    std::vector<OutT> actual(C * OH * OW, 0);
    ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(OutT), satDev, actual.size() * sizeof(OutT),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "copy sat back failed: " << ret << std::endl; return ret);

    bool ok = true;
    for (int64_t i = 0; i < C * OH * OW; i++) {
        double got = static_cast<double>(actual[i]);
        double want = expected[i];
        double tol = (std::fabs(want) > 1.0) ? std::fabs(want) * 1e-4 : 1e-3;
        if (std::fabs(got - want) > tol) {
            ok = false;
            std::cout << "sat3d mismatch at " << i << ": got " << got << " expected " << want << std::endl;
            break;
        }
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(imageTensor);
    aclDestroyTensor(satTensor);
    aclrtFree(imageDev);
    aclrtFree(satDev);

    std::cout << (ok ? "[PASS]" : "[FAIL]") << " IntegralImage 3D " << typeid(InT).name() << " C=" << C << " H=" << H
              << " W=" << W << std::endl;
    return ok ? 0 : 1;
}

// ---------------- benchmark ----------------
// 与 torch_npu 基线同口径：warmup 3 次，迭代 iters 次，取中位数（微秒）。
// 计时区域 = aclnnIntegralImage 下发 + aclrtSynchronizeStream 等待完成。
template <typename InT, typename OutT>
double BenchShape(aclrtStream stream, int64_t H, int64_t W, aclDataType inAclType, aclDataType outAclType, int iters)
{
    const int64_t OH = H + 1;
    const int64_t OW = W + 1;

    // 伪随机输入 0..255（与 torch 基线 randint(0,256) 对齐）
    std::vector<InT> input(H * W);
    unsigned int seed = 12345u;
    for (auto& v : input) {
        seed = seed * 1103515245u + 12345u;
        v = static_cast<InT>((seed >> 16) & 0xFF);
    }

    aclTensor* imageTensor = nullptr;
    aclTensor* satTensor = nullptr;
    void* imageDev = nullptr;
    void* satDev = nullptr;
    std::vector<int64_t> inShape = {H, W};
    std::vector<int64_t> outShape = {OH, OW};

    int ret = CreateAclTensor(input, inShape, &imageDev, inAclType, &imageTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: create image tensor failed" << std::endl; return -1.0);
    ret = CreateAclTensor(std::vector<OutT>(), outShape, &satDev, outAclType, &satTensor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: create sat tensor failed" << std::endl; return -1.0);

    const int warmup = 3;
    for (int i = 0; i < warmup; i++) {
        uint64_t wsSizeTmp = 0;
        aclOpExecutor* exTmp = nullptr;
        ret = aclnnIntegralImageGetWorkspaceSize(imageTensor, satTensor, &wsSizeTmp, &exTmp);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: GetWorkspaceSize failed: " << ret << std::endl; return -1.0);
        void* wsTmp = nullptr;
        if (wsSizeTmp > 0) {
            ret = aclrtMalloc(&wsTmp, wsSizeTmp, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: workspace malloc failed" << std::endl; return -1.0);
        }
        ret = aclnnIntegralImage(wsTmp, wsSizeTmp, exTmp, stream);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: aclnnIntegralImage failed: " << ret << std::endl;
                  return -1.0);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: sync stream failed: " << ret << std::endl; return -1.0);
        if (wsSizeTmp > 0) {
            aclrtFree(wsTmp);
        }
    }

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: sync stream failed: " << ret << std::endl; return -1.0);
        uint64_t wsSizeTmp = 0;
        aclOpExecutor* exTmp = nullptr;
        ret = aclnnIntegralImageGetWorkspaceSize(imageTensor, satTensor, &wsSizeTmp, &exTmp);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: GetWorkspaceSize failed: " << ret << std::endl; return -1.0);
        void* wsTmp = nullptr;
        if (wsSizeTmp > 0) {
            ret = aclrtMalloc(&wsTmp, wsSizeTmp, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: workspace malloc failed" << std::endl; return -1.0);
        }
        auto t0 = std::chrono::steady_clock::now();
        ret = aclnnIntegralImage(wsTmp, wsSizeTmp, exTmp, stream);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: aclnnIntegralImage failed: " << ret << std::endl;
                  return -1.0);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "bench: sync stream failed: " << ret << std::endl; return -1.0);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (wsSizeTmp > 0) {
            aclrtFree(wsTmp);
        }
    }

    std::sort(times.begin(), times.end());
    double med = times[iters / 2];
    std::cout << "op integral_image " << typeid(InT).name() << "->" << typeid(OutT).name() << " H=" << H << " W=" << W
              << ": median=" << med << " us (min=" << times.front() << ", max=" << times.back() << ", n=" << iters
              << ")" << std::endl;

    aclDestroyTensor(imageTensor);
    aclDestroyTensor(satTensor);
    aclrtFree(imageDev);
    aclrtFree(satDev);
    return med;
}

int RunBench(aclrtStream stream, int iters)
{
    // 主口径：uint8 -> int32（与 torch_npu cumsum int32 基线同输出类型、同尺寸）
    int64_t sizes[][2] = {{1024, 1024}, {2048, 2048}, {4096, 4096}};
    for (auto& s : sizes) {
        double t = BenchShape<uint8_t, int32_t>(stream, s[0], s[1], ACL_UINT8, ACL_INT32, iters);
        if (t < 0) {
            return 1;
        }
    }
    // 附：fp32 -> fp32 同尺寸（可另与 torch cumsum fp32 口径对比）
    for (auto& s : sizes) {
        double t = BenchShape<float, float>(stream, s[0], s[1], ACL_FLOAT, ACL_FLOAT, iters);
        if (t < 0) {
            return 1;
        }
    }
    return 0;
}

int RunAll(aclrtStream stream)
{
    int ret = 0;

    // 用例 1：uint8 8x64 全 1（单核），期望 sat[i][j]=i*j，验证零填充边
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 64;
        std::vector<uint8_t> in(H * W, 1);
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = static_cast<double>(i) * j;
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

#ifndef DUMP_SAT
    // 用例 2：uint8 8x384 伪随机（3 核列分块），验证块起始补偿
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 384;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 7 + 3) % 97);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 3：float16 8x128 全 1.0（两级转换 fp16 -> fp32）
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 128;
        std::vector<uint16_t> in(H * W, FloatToHalf(1.0f));
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = static_cast<double>(i) * j;
            }
        }
        ret |= RunCaseT<uint16_t, float>(stream, H, W, in, ACL_FLOAT16, ACL_FLOAT, exp);
    }

    // 用例 4：float32 8x128 伪随机（fp32 -> fp32）
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 128;
        std::vector<float> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<float>((i * 3 % 17)) * 0.25f;
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<float, float>(stream, H, W, in, ACL_FLOAT, ACL_FLOAT, exp);
    }

    // 用例 5：float32 3D 2x8x128（HWC 逐通道独立）
    {
        constexpr int64_t C = 2;
        constexpr int64_t H = 8;
        constexpr int64_t W = 128;
        std::vector<float> in(C * H * W);
        for (int64_t i = 0; i < C * H * W; i++) {
            in[i] = static_cast<float>((i * 13 + 5) % 89);
        }
        std::vector<double> exp((H + 1) * (W + 1) * C, 0);
        for (int64_t c = 0; c < C; c++) {
            for (int64_t i = 1; i <= H; i++) {
                for (int64_t j = 1; j <= W; j++) {
                    // HWC 索引：idx = i*(W+1)*C + j*C + c
                    int64_t idx = i * (W + 1) * C + j * C + c;
                    exp[idx] = exp[(i - 1) * (W + 1) * C + j * C + c] +
                               exp[i * (W + 1) * C + (j - 1) * C + c] -
                               exp[(i - 1) * (W + 1) * C + (j - 1) * C + c] +
                               in[(i - 1) * W * C + (j - 1) * C + c];
                }
            }
        }
        ret |= RunCase3DT<float, float>(stream, C, H, W, in, ACL_FLOAT, ACL_FLOAT, exp);
    }

    // 用例 6：uint8 1x1（W<32 全尾部，DataCopyPad 路径）
    {
        constexpr int64_t H = 1;
        constexpr int64_t W = 1;
        std::vector<uint8_t> in(H * W, 7);
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = static_cast<double>(i) * j * 7;
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 7：uint8 1x64（单行）
    {
        constexpr int64_t H = 1;
        constexpr int64_t W = 64;
        std::vector<uint8_t> in(H * W, 3);
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = static_cast<double>(i) * j * 3;
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 8：uint8 8x1（单列，W<32）
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 1;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>(i + 1);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 9：uint8 257x513（大非对齐，单核全 DataCopyPad）
    {
        constexpr int64_t H = 257;
        constexpr int64_t W = 513;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 5 + 1) % 251);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 10：uint8 8x2048（真多核：blockW=1024 => 2 核）
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 2048;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 3 + 11) % 233);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // 用例 11：uint8 8x4097（4 核 + 尾部 1 列，多核非对齐）
    {
        constexpr int64_t H = 8;
        constexpr int64_t W = 4097;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 7 + 5) % 199);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // Case 12: uint8 256x256 large two-phase smoke
    {
        constexpr int64_t H = 256;
        constexpr int64_t W = 256;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 11 + 7) % 173);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // Case 13: uint8 512x512 common size (same as torch baseline 512x512 path)
    {
        constexpr int64_t H = 512;
        constexpr int64_t W = 512;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 17 + 3) % 251);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // Case 14: uint8 1x4096 single row (two-phase, only core0 does horizontal scan)
    {
        constexpr int64_t H = 1;
        constexpr int64_t W = 4096;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 3 + 5) % 199);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // Case 15: uint8 4096x1 single column (W<128 legacy single-core path)
    {
        constexpr int64_t H = 4096;
        constexpr int64_t W = 1;
        std::vector<uint8_t> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<uint8_t>((i * 5 + 1) % 233);
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<uint8_t, int32_t>(stream, H, W, in, ACL_UINT8, ACL_INT32, exp);
    }

    // Case 16: float32 480x640 non-standard resolution
    {
        constexpr int64_t H = 480;
        constexpr int64_t W = 640;
        std::vector<float> in(H * W);
        for (int64_t i = 0; i < H * W; i++) {
            in[i] = static_cast<float>((i * 7 % 97)) * 0.125f;
        }
        std::vector<double> exp((H + 1) * (W + 1), 0);
        for (int64_t i = 1; i <= H; i++) {
            for (int64_t j = 1; j <= W; j++) {
                exp[i * (W + 1) + j] = exp[(i - 1) * (W + 1) + j] + exp[i * (W + 1) + j - 1] -
                                       exp[(i - 1) * (W + 1) + j - 1] + in[(i - 1) * W + (j - 1)];
            }
        }
        ret |= RunCaseT<float, float>(stream, H, W, in, ACL_FLOAT, ACL_FLOAT, exp);
    }

    return ret;
#endif
}

int main(int argc, char* argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init acl failed: " << ret << std::endl; return ret);

    bool benchMode = (argc >= 2) && (std::string(argv[1]) == "bench");
    if (benchMode) {
        int iters = (argc >= 3) ? std::atoi(argv[2]) : 20;
        if (iters <= 0) {
            iters = 20;
        }
        std::cout << "==== IntegralImage benchmark mode, iters=" << iters << " ====" << std::endl;
        ret = RunBench(stream, iters);
    } else {
        ret = RunAll(stream);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
