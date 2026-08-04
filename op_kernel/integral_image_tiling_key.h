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
 * \file integral_image_tiling_key.h
 * \brief IntegralImage tiling key declare
 */

#ifndef __INTEGRAL_IMAGE_TILING_KEY_H__
#define __INTEGRAL_IMAGE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 0: uint8   -> INT32 累加
// 1: float16 -> FP32 累加
// 2: float32 -> FP32 累加
#define INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32 0
#define INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32 1
#define INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32 2

ASCENDC_TPL_ARGS_DECL(IntegralImage, ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                                                           INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32,
                                                           INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32,
                                                           INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32,
                                                          INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32,
                                                          INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32)));

#endif // __INTEGRAL_IMAGE_TILING_KEY_H__
