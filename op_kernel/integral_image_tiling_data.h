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
 * \file integral_image_tiling_data.h
 * \brief IntegralImage tiling data struct
 */

#ifndef _INTEGRAL_IMAGE_TILING_DATA_H_
#define _INTEGRAL_IMAGE_TILING_DATA_H_

struct IntegralImageTilingData {
    int64_t height = 0;      // 输入 H
    int64_t width = 0;       // 输入 W
    int64_t channel = 1;     // 输入 C（3D），2D 时为 1
    int64_t blockWidth = 0;  // 每核处理的列宽（32 对齐）
    int64_t coreNum = 1;     // 使用的 AI Core 数
};

#endif // _INTEGRAL_IMAGE_TILING_DATA_H_
