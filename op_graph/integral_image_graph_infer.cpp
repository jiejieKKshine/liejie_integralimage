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
 * \file integral_image_graph_infer.cpp
 * \brief IntegralImage graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 推断输出数据类型：整数输入（uint8/int16）-> int32，浮点输入（float16/float32）-> float32
 */
static ge::graphStatus InferDataTypeIntegralImage(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeIntegralImage");

    ge::DataType inDtype = context->GetInputDataType(IDX_0);
    ge::DataType outDtype = ge::DT_INT32;
    if (inDtype == ge::DT_FLOAT16 || inDtype == ge::DT_FLOAT) {
        outDtype = ge::DT_FLOAT;
    }
    context->SetOutputDataType(IDX_0, outDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeIntegralImage");
    return GRAPH_SUCCESS;
}

IMPL_OP(IntegralImage).InferDataType(InferDataTypeIntegralImage);

} // namespace ops
