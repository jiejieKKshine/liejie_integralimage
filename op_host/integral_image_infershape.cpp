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
 * \file integral_image_infershape.cpp
 * \brief IntegralImage 输出 shape 推导
 *
 * 输出 sat 与输入 image 的 shape 关系（HWC 布局）：
 *   (H,W)   -> (H+1,W+1)
 *   (H,W,C) -> (H+1,W+1,C)
 * 即第 0/1 维（H、W）各 +1（物理零填充边），C 维不变。
 *
 * 约束（与 Tiling 一致）：
 *   - rank 必须为 2 或 3
 *   - 动态 rank（-2）透传为 UnknownRank
 *   - 动态维度（-1）：最后两维保持未知，不执行 +1
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeIntegralImage(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeIntegralImage");

    const gert::Shape* inputShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    gert::Shape* outputShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 动态 rank（-2）：输出透传 UnknownRank
    if (Ops::Base::IsUnknownRank(*inputShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank.");
        Ops::Base::SetUnknownRank(*outputShape);
        return GRAPH_SUCCESS;
    }

    auto dimNum = inputShape->GetDimNum();
    if (dimNum != 2 && dimNum != 3) {
        OP_LOGE(context, "IntegralImage requires rank 2 or 3, got %zu", dimNum);
        return GRAPH_FAILED;
    }
    outputShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        int64_t dim = inputShape->GetDim(i);
        // 第 0/1 维是 H、W，各 +1（HWC：C 在最后一维，不变）；动态维度保持未知
        if ((i == 0 || i == 1) && dim != ge::UNKNOWN_DIM) {
            dim += 1;
        }
        outputShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeIntegralImage");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IntegralImage).InferShape(InferShapeIntegralImage);

} // namespace ops
