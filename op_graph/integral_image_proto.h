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
 * \file integral_image_proto.h
 * \brief IntegralImage 算子原型定义
 */
#ifndef OPS_OP_PROTO_INC_INTEGRALIMAGE_H_
#define OPS_OP_PROTO_INC_INTEGRALIMAGE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the integral image (Summed-Area Table) of an input image.
*@par Inputs:
*One input, including:
* @li image: A ND Tensor. Must be one of the following types: uint8, float16, float32.
*            Shape (H,W) or (H,W,C). For 3D input, the integral image is computed
*            independently per channel (C is not reduced).
*@par Outputs:
*sat: A ND Tensor. Must be one of the following types: int32, float32.
*     Shape (H+1,W+1) or (H+1,W+1,C) with a physical zero-padded top row and
*     left column (OpenCV convention): sat[0][*]=sat[*][0]=0.
*@par Attributes:
*sdepth: An optional int32, output sum depth selector (OpenCV sdepth semantics):
*        -1 (default) auto: uint8->int32, float16/float32->float32;
*        0 = int32 (uint8 input only); 1 = float32 (all supported inputs);
*        2 = float64 (not supported yet on Ascend 910B).
*@par Third-party framework compatibility
*Compatible with OpenCV cv::integral / NVIDIA NPP nppiIntegral / MATLAB integralImage.
*/
REG_OP(IntegralImage)
    .INPUT(image, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sat, TensorType({DT_INT32, DT_FLOAT, DT_FLOAT}))
    .ATTR(sdepth, Int, -1)
    .OP_END_FACTORY_REG(IntegralImage)

} // namespace ge

#endif // OPS_OP_PROTO_INC_INTEGRALIMAGE_H_
