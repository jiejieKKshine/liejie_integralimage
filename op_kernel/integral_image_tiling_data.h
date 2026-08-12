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
    int64_t height = 0;      // input H
    int64_t width = 0;       // input W
    int64_t channel = 1;     // input C (3D only), 1 for 2D
    int64_t blockWidth = 0;  // column block width for legacy path (32 aligned)
    int64_t coreNum = 1;     // block dim (number of AI cores launched)
    // ---- two-phase: per-row horizontal scan + workspace + per-column vertical add ----
    int64_t twoPhase = 0;        // 1 = enable two-phase (2D and W >= 128)
    int64_t activeCoreNum = 1;   // cores used by two-phase: min(platform AIV, max(H, colTileCount))
    int64_t rowTileWidth = 0;    // phase A in-row tile width (UB bound, <= 1024)
    int64_t colTileWidth = 0;    // phase B column tile width (vector width)
    int64_t colTileCount = 0;    // phase B column tile count = ceil(W / colTileWidth)
    int64_t sqsumEnabled = 1;    // optional sqsum output provided by the caller
    int64_t tiltedEnabled = 1;   // optional tilted output provided by the caller
};

#endif // _INTEGRAL_IMAGE_TILING_DATA_H_
