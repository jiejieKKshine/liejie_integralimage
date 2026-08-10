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
 * \file integral_image.cpp
 * \brief IntegralImage kernel entry
 */

#include "integral_image.h"
#include "integral_image_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void integral_image(
    GM_ADDR image, GM_ADDR sat, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(IntegralImageTilingData);
    GET_TILING_DATA_WITH_STRUCT(IntegralImageTilingData, tilingData, tiling);

    if constexpr (schMode == INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32) {
        NsIntegralImage::IntegralImage<uint8_t, int32_t> op;
        op.Init(image, sat, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32) {
        NsIntegralImage::IntegralImage<half, float> op;
        op.Init(image, sat, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32) {
        NsIntegralImage::IntegralImage<float, float> op;
        op.Init(image, sat, workspace, &tilingData);
        op.Process();
    } else {
        // INTEGRAL_IMAGE_TPL_SCH_MODE_U8_F32: uint8 input, float32 output (sdepth=1)
        NsIntegralImage::IntegralImage<uint8_t, float> op;
        op.Init(image, sat, workspace, &tilingData);
        op.Process();
    }
}
