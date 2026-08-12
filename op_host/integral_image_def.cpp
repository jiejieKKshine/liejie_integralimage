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
 * \file integral_image_def.cpp
 * \brief IntegralImage 算子定义
 */

#include "register/op_def_registry.h"

namespace ops {

class IntegralImage : public OpDef {
public:
    explicit IntegralImage(const char* name) : OpDef(name)
    {
        this->Input("image")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("sat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // sqsum: 像素平方积分图（OpenCV integral2 语义），float32 输出。
        // 910B 向量单元不支持 double，故用 fp32 近似 OpenCV 默认 CV_64F。
        // OPTIONAL 输出：调用方不传（空 tensor）时 kernel 跳过 sqsum 计算，
        // 与 OpenCV integral（只 sum）形态对齐。
        this->Output("sqsum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // tilted: 45° 旋转积分图（OpenCV integral3 语义），输出类型与 sat 一致。
        // 采用对角前缀分解的多核实现（主/反对角线各自独立）。
        // OPTIONAL 输出：调用方不传（空 tensor）时 kernel 跳过 tilted 计算，主算子两路性能不受影响。
        this->Output("tilted")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // sdepth: 输出深度选择（OpenCV 语义）。-1 自动（u8->int32，fp16/fp32->float32）；
        // 0=int32（仅 uint8 输入）；1=float32；2=float64（910B 暂不支持，tiling 校验拒绝）。
        this->Attr("sdepth").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "integral_image");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(IntegralImage);

} // namespace ops
