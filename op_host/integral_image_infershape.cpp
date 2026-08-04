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
 * 输出 sat 与输入 image 的 shape 关系：
 *   (H,W)   -> (H+1,W+1)
 *   (H,W,C) -> (H+1,W+1,C)
 * 即最后两维各 +1（物理零填充边），更高维不变。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

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

    auto dimNum = inputShape->GetDimNum();
    if (dimNum < 2) {
        OP_LOGE(context, "IntegralImage requires rank >= 2, got %zu", dimNum);
        return GRAPH_FAILED;
    }
    outputShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        int64_t dim = inputShape->GetDim(i);
        // 最后两维是 H、W，各 +1
        if (i == dimNum - 1 || i == dimNum - 2) {
            dim += 1;
        }
        outputShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeIntegralImage");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IntegralImage).InferShape(InferShapeIntegralImage);

} // namespace ops
