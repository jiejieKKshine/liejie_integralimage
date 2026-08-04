/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file integral_image_tiling.cpp
 * \brief IntegralImage host tiling：按列分块多核
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/integral_image_tiling_data.h"
#include "../op_kernel/integral_image_tiling_key.h"

namespace optiling {

constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr int32_t MAX_BLOCK_WIDTH = 1024;
constexpr int32_t MIN_BLOCK_WIDTH = 32;

struct IntegralImageCompileInfo {
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IntegralImageTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNumAiv;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNumAiv) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(INDEXZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inputShapeX = inputShape->GetStorageShape();
    auto dimNum = inputShapeX.GetDimNum();
    if (dimNum != 2 && dimNum != 3) {
        OP_LOGE(context, "IntegralImage requires rank 2 or 3, got %zu", dimNum);
        return ge::GRAPH_FAILED;
    }

    // HWC 布局：rank2=(H,W)，rank3=(H,W,C)，C 在最后一维
    int64_t height = inputShapeX.GetDim(INDEXZERO);
    int64_t width = inputShapeX.GetDim(INDEXONE);
    int64_t channel = (dimNum == 3) ? inputShapeX.GetDim(dimNum - 1) : 1;
    if (height <= 0 || width <= 0 || channel <= 0) {
        OP_LOGE(context, "invalid shape H=%ld W=%ld C=%ld", height, width, channel);
        return ge::GRAPH_FAILED;
    }
    auto inDtype = context->GetInputDesc(INDEXZERO)->GetDataType();
    // blockWidth 受 UB 约束：整块加载 blockW*C（InT + AccT 写回 + C*blockW 累加器）
    // 预算取 ubSize 的 70%，留余量给临时缓冲
    int64_t inBytes = (inDtype == ge::DT_UINT8) ? 1 : ((inDtype == ge::DT_FLOAT16) ? 2 : 4);
    const int64_t accBytes = 4;
    int64_t ubBudget = static_cast<int64_t>(ubSize) * 7 / 10;
    int32_t blockWidth = MAX_BLOCK_WIDTH;
    while (blockWidth > MIN_BLOCK_WIDTH &&
           blockWidth * channel * (inBytes + 2 * accBytes) + 3 * blockWidth * accBytes > ubBudget) {
        blockWidth >>= 1;
    }
    // 核数向上取整：最后一个核处理非 32 对齐的尾部（W<32 时单核全尾部）
    int32_t coreNum = static_cast<int32_t>((width + blockWidth - 1) / blockWidth);
    // 限制到平台实际 AI Core 数；当前 kernel 每核单块，
    // 若块数超过平台核数（W > coreNum*blockW）则超当前支持范围，报错
    if (coreNum > coreNumAiv) {
        OP_LOGE(context, "IntegralImage: W=%ld requires %d blocks > platform cores %ld (per-core multi-block not implemented)",
                width, coreNum, coreNumAiv);
        return ge::GRAPH_FAILED;
    }

    IntegralImageTilingData* tiling = context->GetTilingData<IntegralImageTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(IntegralImageTilingData), 0, sizeof(IntegralImageTilingData)) != EOK,
                OP_LOGE(context, "memset tiling error"), return ge::GRAPH_FAILED);
    tiling->height = height;
    tiling->width = width;
    tiling->channel = channel;
    tiling->blockWidth = blockWidth;
    tiling->coreNum = coreNum;

    context->SetBlockDim(coreNum);

    // 按输入 dtype 分派 tiling key（3 组合：uint8/float16/float32）
    uint64_t tilingKey = 0;
    if (inDtype == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32);
    } else if (inDtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32);
    } else if (inDtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32);
    } else {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIntegralImage([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IntegralImage)
    .Tiling(IntegralImageTilingFunc)
    .TilingParse<IntegralImageCompileInfo>(TilingParseForIntegralImage);

} // namespace optiling
