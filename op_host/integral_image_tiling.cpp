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
 * \brief IntegralImage host tiling: two-phase (per-row scan + per-column vertical add) or legacy column blocks
 */

#include "log/log.h"
#include <algorithm>
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
constexpr int32_t TWO_PHASE_MIN_WIDTH = 128;
constexpr int32_t MAX_COL_TILE_WIDTH = 256;
constexpr int32_t MIN_COL_TILE_WIDTH = 32;
// Platform (Ascend 910B, NPU_ARCH 2201) reserves the first 16MB of the caller workspace
// for the runtime (kernel_utils_constants.h: RESERVED_WORKSPACE). The kernel's workspace
// GM_ADDR points to base + 16MB, so the tiling must request reserved + user size.
constexpr int64_t RESERVED_WORKSPACE_BYTES = 16LL * 1024 * 1024;

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

    // HWC layout: rank2 = (H, W), rank3 = (H, W, C), C is the last dim.
    int64_t height = inputShapeX.GetDim(INDEXZERO);
    int64_t width = inputShapeX.GetDim(INDEXONE);
    int64_t channel = (dimNum == 3) ? inputShapeX.GetDim(dimNum - 1) : 1;
    if (height <= 0 || width <= 0 || channel <= 0) {
        OP_LOGE(context, "invalid shape H=%ld W=%ld C=%ld", height, width, channel);
        return ge::GRAPH_FAILED;
    }
    auto inDtype = context->GetInputDesc(INDEXZERO)->GetDataType();
    // Optional outputs: disabled when the caller provides an empty tensor (rank 0 or zero dim).
    int64_t sqsumEnabled = 1;
    auto sqsumShape = context->GetOutputShape(1);
    if (sqsumShape == nullptr) {
        sqsumEnabled = 0;
    } else {
        auto sq = sqsumShape->GetShape();
        if (sq.GetDimNum() == 0 || (sq.GetDimNum() == 1 && sq.GetDim(0) == 0)) {
            sqsumEnabled = 0;
        }
    }
    int64_t tiltedEnabled = 1;
    auto tiltedShape = context->GetOutputShape(2);
    if (tiltedShape == nullptr) {
        tiltedEnabled = 0;
    } else {
        auto shp = tiltedShape->GetShape();
        if (shp.GetDimNum() == 0 || (shp.GetDimNum() == 1 && shp.GetDim(0) == 0)) {
            tiltedEnabled = 0;
        }
    }
    // sdepth（OpenCV 语义）：-1 自动（u8->int32，fp16/fp32->float32）；0=int32；1=float32；
    // 2=float64（910B 向量单元不支持 double，暂拒绝）。
    int64_t sdepth = -1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            sdepth = *p;
        }
    }
    if (sdepth != -1 && sdepth != 0 && sdepth != 1) {
        OP_LOGE(context, "IntegralImage: unsupported sdepth=%ld (expect -1/0/1; float64 not supported on 910B)",
                sdepth);
        return ge::GRAPH_FAILED;
    }
    if (sdepth == 0 && inDtype != ge::DT_UINT8) {
        OP_LOGE(context, "IntegralImage: sdepth=int32 is only valid for uint8 input");
        return ge::GRAPH_FAILED;
    }
    const bool outF32 = (sdepth == 1);
    const int64_t accBytes = 4;
    int64_t inBytes = (inDtype == ge::DT_UINT8) ? 1 : ((inDtype == ge::DT_FLOAT16) ? 2 : 4);

    IntegralImageTilingData* tiling = context->GetTilingData<IntegralImageTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(IntegralImageTilingData), 0, sizeof(IntegralImageTilingData)) != EOK,
                OP_LOGE(context, "memset tiling error"), return ge::GRAPH_FAILED);
    tiling->height = height;
    tiling->width = width;
    tiling->channel = channel;
    tiling->sqsumEnabled = sqsumEnabled;
    tiling->tiltedEnabled = tiltedEnabled;

    if (channel == 1 && width >= TWO_PHASE_MIN_WIDTH) {
        // ---- two-phase path: phase A per-row horizontal scan, phase B per-column vertical add ----
        // UB budget must also reserve the phase-A batch buffers and the CumSum LCM tmp
        // (TPipe allocates all UB positions from one pool; PopStackBuffer(LCM) takes the tail).
        // fixed UB = legacy buffers (~90KB with sqsum phase-B additions):
        // wsBatch/out/wsSqBatch/outSq 8*(colTileW+8)*4*4 + rows/cols/acc/zero/left/cast etc.
        const int64_t kFixedUb = 96 * 1024;
        const bool fpPath = (inDtype == ge::DT_FLOAT16 || inDtype == ge::DT_FLOAT);
        int64_t rowTileWidth = MAX_BLOCK_WIDTH;
        int64_t ubBudget = static_cast<int64_t>(ubSize) * 9 / 10;
        while (rowTileWidth > MIN_BLOCK_WIDTH) {
            int64_t batchBytes = 0;
            if (fpPath) {
                int64_t extraCast = (inDtype == ge::DT_FLOAT16) ? accBytes : 0;
                batchBytes = 8LL * (rowTileWidth + 8) * (inBytes + accBytes + extraCast) +
                             8LL * rowTileWidth * accBytes * 2 + // CumSum LCM tmp
                             8LL * rowTileWidth * accBytes * 2 + // sqsum: sqSrcBatch + dstSqBatch
                             rowTileWidth * accBytes;            // sqsum column accumulator (accSq)
            } else {
                // u8: imgBatch(InT) + dstBatch + dstSqBatch(float32) + sqsum accumulator
                batchBytes = 8LL * (rowTileWidth + 8) * (inBytes + 2 * accBytes) + rowTileWidth * accBytes;
            }
            if (kFixedUb + batchBytes <= ubBudget) {
                break;
            }
            rowTileWidth >>= 1;
        }
        // Phase B: one column tile per core for wide images. colTileWidth is chosen so that
        // colTileCount is close to (but not exceeding) the core count, keeping the vector
        // width >= 32 elements. Too many tiles would make the busiest core run 2+ tiles serially.
        int64_t perCore = (width + coreNumAiv - 1) / coreNumAiv;
        int64_t colTileWidth = ((perCore + 7) / 8) * 8; // 32B aligned for AccT=float
        if (colTileWidth < MIN_COL_TILE_WIDTH) {
            colTileWidth = MIN_COL_TILE_WIDTH;
        }
        if (colTileWidth > MAX_COL_TILE_WIDTH) {
            colTileWidth = MAX_COL_TILE_WIDTH;
        }
        if (colTileWidth > width) {
            colTileWidth = width;
        }
        int64_t colTileCount = (width + colTileWidth - 1) / colTileWidth;
        int64_t activeCoreNum = std::min(coreNumAiv, std::max(height, colTileCount));
        if (activeCoreNum < 1) {
            activeCoreNum = 1;
        }

        tiling->twoPhase = 1;
        tiling->activeCoreNum = activeCoreNum;
        tiling->rowTileWidth = rowTileWidth;
        tiling->colTileWidth = colTileWidth;
        tiling->colTileCount = colTileCount;
        tiling->blockWidth = rowTileWidth; // reused by kernel UB buffer sizing
        tiling->coreNum = activeCoreNum;

        context->SetBlockDim(static_cast<uint32_t>(activeCoreNum));

        size_t* ws = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        // workspace layout:
        //   [0, H*W)        sum row prefixes (AccT)
        //   [H*W, 2*H*W)    sqsum row prefixes (float)
        //   [2*H*W, 4*H*W)  tilted acc1 / acc2 diagonal-prefix buffers (AccT)
        ws[0] = static_cast<size_t>(RESERVED_WORKSPACE_BYTES + height * width * accBytes * 4);
    } else {
        // ---- legacy path: per-column block multi-core (2D small W or 3D HWC) ----
        int64_t ubBudget = static_cast<int64_t>(ubSize) * 7 / 10;
        int32_t blockWidth = MAX_BLOCK_WIDTH;
        while (blockWidth > MIN_BLOCK_WIDTH &&
               blockWidth * channel * (inBytes + 2 * accBytes) + 3 * blockWidth * accBytes > ubBudget) {
            blockWidth >>= 1;
        }
        int32_t coreNum = static_cast<int32_t>((width + blockWidth - 1) / blockWidth);
        if (coreNum > coreNumAiv) {
            OP_LOGE(context,
                    "IntegralImage: W=%ld requires %d blocks > platform cores %ld (per-core multi-block not implemented)",
                    width, coreNum, coreNumAiv);
            return ge::GRAPH_FAILED;
        }

        tiling->blockWidth = blockWidth;
        tiling->coreNum = coreNum;
        context->SetBlockDim(static_cast<uint32_t>(coreNum));
        // tilted multi-core needs workspace: [0,H*W) row prefixes, [H*W,2*H*W) acc1,
        // [2*H*W,3*H*W) acc2
        size_t* ws = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        ws[0] = static_cast<size_t>(RESERVED_WORKSPACE_BYTES + height * width * accBytes * 3);
    }

    uint64_t tilingKey = 0;
    if (inDtype == ge::DT_UINT8) {
        tilingKey = outF32 ? GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_F32)
                           : GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32);
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
