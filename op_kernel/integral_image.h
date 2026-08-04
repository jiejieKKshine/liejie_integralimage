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
 * \file integral_image.h
 * \brief IntegralImage kernel 实现（Ascend C）
 *
 * v4 设计（HWC 布局，对齐需求表）：
 *   - 输入 image: (H,W) 或 (H,W,C)，uint8/float16/float32，ND
 *   - 输出 sat  : (H+1,W+1) 或 (H+1,W+1,C)，int32（整数输入）/ float32（浮点输入）
 *   - HWC 下每通道平面跨步存储（相邻 w 间隔 C 元素），因此每行整块加载
 *     blockW*C 连续元素，在 UB 内按通道提取（步长 C）处理，再交错写回。
 *   - 每通道独立列累加器（accBig_），多核按列分块 + 块起始补偿。
 *   - 物理零填充边：sat[0][*]=sat[*][0]=0（OpenCV 惯例）。
 */
#ifndef INTEGRAL_IMAGE_H
#define INTEGRAL_IMAGE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "integral_image_tiling_data.h"

namespace NsIntegralImage {

using namespace AscendC;

template <typename InT, typename AccT>
class IntegralImage {
public:
    __aicore__ inline IntegralImage() {}
    __aicore__ inline void Init(GM_ADDR image, GM_ADDR sat, const IntegralImageTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void RowScan(LocalTensor<AccT>& row, int32_t len);
    __aicore__ inline void ExtractChannel(LocalTensor<AccT>& dst, const LocalTensor<InT>& src, int32_t c,
                                          int32_t channel, int32_t len);
    __aicore__ inline AccT SumChannel(const LocalTensor<InT>& src, int32_t c, int32_t channel, int32_t len);
    __aicore__ inline void ScatterChannel(LocalTensor<AccT>& dst, const LocalTensor<AccT>& src, int32_t c,
                                          int32_t channel, int32_t len);

private:
    TPipe pipe_;
    // HWC 整块缓冲（交错）：imgBuf_=blockW*C InT，satBuf_=blockW*C AccT
    TBuf<TPosition::VECIN> imgBuf_;
    TBuf<TPosition::VECIN> satBuf_;
    TBuf<TPosition::VECIN> accBigBuf_;   // C*blockW AccT，每通道列累加器
    TBuf<TPosition::VECIN> prefixBuf_;   // C AccT，每通道块起始补偿
    TBuf<TPosition::VECIN> zeroBigBuf_;  // 8*C AccT，零填充边界（独立缓冲，避免污染 satBuf）
    TBuf<TPosition::VECIN> castBuf_;
    TBuf<TPosition::VECIN> zeroBuf_;

    LocalTensor<InT> imgLocal_;
    LocalTensor<AccT> satLocal_;
    LocalTensor<AccT> accBigLocal_;
    LocalTensor<AccT> prefixLocal_;
    LocalTensor<AccT> zeroBigLocal_;
    LocalTensor<AccT> castLocal_;
    LocalTensor<AccT> zeroLocal_;

    GlobalTensor<InT> imageGm_;
    GlobalTensor<AccT> satGm_;

    int64_t height_ = 0;
    int64_t width_ = 0;
    int64_t channel_ = 1;
    int32_t blockWidth_ = 0;
    int32_t colStart_ = 0;
    int32_t evMte2V_ = 0;
    int32_t evMte2S_ = 0;
    int32_t evSMte2_ = 0;
    int32_t evVS_ = 0;
    int32_t evSV_ = 0;
    int32_t evSMte3_ = 0;
    int32_t evMte3V_ = 0;
};

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::Init(GM_ADDR image, GM_ADDR sat,
    const IntegralImageTilingData* tilingData)
{
    height_ = tilingData->height;
    width_ = tilingData->width;
    channel_ = tilingData->channel;
    blockWidth_ = static_cast<int32_t>(tilingData->blockWidth);
    colStart_ = AscendC::GetBlockIdx() * blockWidth_;

    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);

    imageGm_.SetGlobalBuffer((__gm__ InT*)image);
    satGm_.SetGlobalBuffer((__gm__ AccT*)sat);

    pipe_.InitBuffer(imgBuf_, blockElems * sizeof(InT));
    pipe_.InitBuffer(satBuf_, blockElems * sizeof(AccT));
    pipe_.InitBuffer(accBigBuf_, blockElems * sizeof(AccT));
    pipe_.InitBuffer(prefixBuf_, static_cast<int32_t>(channel_) * sizeof(AccT));
    pipe_.InitBuffer(zeroBigBuf_, 8 * static_cast<int32_t>(channel_) * sizeof(AccT));
    pipe_.InitBuffer(castBuf_, blockWidth_ * sizeof(AccT));
    pipe_.InitBuffer(zeroBuf_, blockWidth_ * sizeof(AccT));

    imgLocal_ = imgBuf_.template Get<InT>();
    satLocal_ = satBuf_.template Get<AccT>();
    accBigLocal_ = accBigBuf_.template Get<AccT>();
    prefixLocal_ = prefixBuf_.template Get<AccT>();
    zeroBigLocal_ = zeroBigBuf_.template Get<AccT>();
    castLocal_ = castBuf_.template Get<AccT>();
    zeroLocal_ = zeroBuf_.template Get<AccT>();

    evMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    evMte2S_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    evSMte2_ = GetTPipePtr()->FetchEventID(HardEvent::S_MTE2);
    evVS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    evSV_ = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    evSMte3_ = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
    evMte3V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::RowScan(LocalTensor<AccT>& row, int32_t len)
{
    AccT running = static_cast<AccT>(0);
    for (int32_t i = 0; i < len; i++) {
        running += row.GetValue(i);
        row.SetValue(i, running);
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::ExtractChannel(LocalTensor<AccT>& dst, const LocalTensor<InT>& src,
    int32_t c, int32_t channel, int32_t len)
{
    for (int32_t i = 0; i < len; i++) {
        dst.SetValue(i, static_cast<AccT>(src.GetValue(c + i * channel)));
    }
}

template <typename InT, typename AccT>
__aicore__ inline AccT IntegralImage<InT, AccT>::SumChannel(const LocalTensor<InT>& src, int32_t c, int32_t channel,
    int32_t len)
{
    AccT sum = static_cast<AccT>(0);
    for (int32_t i = 0; i < len; i++) {
        sum += static_cast<AccT>(src.GetValue(c + i * channel));
    }
    return sum;
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::ScatterChannel(LocalTensor<AccT>& dst, const LocalTensor<AccT>& src,
    int32_t c, int32_t channel, int32_t len)
{
    for (int32_t i = 0; i < len; i++) {
        dst.SetValue(c + i * channel, src.GetValue(i));
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::Process()
{
    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);
    AscendC::Duplicate(zeroLocal_, static_cast<AccT>(0), blockWidth_);
    AscendC::Duplicate(accBigLocal_, static_cast<AccT>(0), blockElems);

    // 第 0 行零填充：每核写自己列块 sat[0][colStart+1 .. colStart+blockW]（偏移 +C）
    // 用 satLocal_（行循环会重新填充，无污染）
    AscendC::Duplicate(satLocal_, static_cast<AccT>(0), blockElems);
    AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
    AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
    AscendC::DataCopy(satGm_[(static_cast<int64_t>(colStart_) + 1) * channel_], satLocal_, blockElems);
    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    // core 0 补 sat[0][0..7][*] = 0（8*C 元素，32B 对齐）
    if (colStart_ == 0) {
        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::DataCopy(satGm_[0], zeroBigLocal_, 8 * static_cast<int32_t>(channel_));
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }

    for (int64_t r = 0; r < height_; r++) {
        // 上一轮 S 标量读 imgLocal_ 完成后方可覆盖
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);

        // 1) 第 0 列零填充（core 0）：写 sat[r+1][0..7][*] = 0，
        //     1..7 列随后被主区域写回覆盖；独立缓冲 zeroBig 避免污染 satBuf
        if (colStart_ == 0) {
            AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
            AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            AscendC::DataCopy(satGm_[(r + 1) * (width_ + 1) * channel_], zeroBigLocal_,
                              8 * static_cast<int32_t>(channel_));
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }

        // 2) 块起始补偿：左侧所有块的列和（每通道独立）。
        //    注意：必须先算左侧块、最后加载主块，避免左侧块加载覆盖主块缓冲。
        AscendC::Duplicate(prefixLocal_, static_cast<AccT>(0), static_cast<int32_t>(channel_));
        for (int64_t off = 0; off < colStart_; off += blockWidth_) {
            AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
            AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            AscendC::DataCopy(imgLocal_, imageGm_[r * width_ * channel_ + off * channel_], blockElems);
            AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
            AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);
            for (int64_t c = 0; c < channel_; c++) {
                prefixLocal_.SetValue(static_cast<int32_t>(c),
                    prefixLocal_.GetValue(static_cast<int32_t>(c)) +
                        SumChannel(imgLocal_, static_cast<int32_t>(c), static_cast<int32_t>(channel_), blockWidth_));
            }
        }

        // 3) 加载主块（HWC 交错，blockW*C 连续）——最后加载，不被左侧块覆盖
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::DataCopy(imgLocal_, imageGm_[r * width_ * channel_ + static_cast<int64_t>(colStart_) * channel_],
                          blockElems);
        AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);

        // 4) 每通道：提取 -> 行前缀和 -> +prefixStart -> 列传播 -> 交错写 satBuf
        for (int64_t c = 0; c < channel_; c++) {
            const int32_t ci = static_cast<int32_t>(c);
            ExtractChannel(castLocal_, imgLocal_, ci, static_cast<int32_t>(channel_), blockWidth_);
            RowScan(castLocal_, blockWidth_);
            // 加块起始补偿（广播）。sat 语义 sat[w]=Σ_{c<w} 由写回偏移 +1 实现
            // （rowPrefix[i] 写到 w=colStart+1+i，即 w 对应 c<w 的累加）。
            const AccT ps = prefixLocal_.GetValue(ci);
            for (int32_t i = 0; i < blockWidth_; i++) {
                castLocal_.SetValue(i, castLocal_.GetValue(i) + ps);
            }
            // 列传播（V 管道，accBig_ 子视图）
            AscendC::SetFlag<HardEvent::S_V>(evSV_);
            AscendC::WaitFlag<HardEvent::S_V>(evSV_);
            AscendC::Add(accBigLocal_[ci * blockWidth_], accBigLocal_[ci * blockWidth_], castLocal_, blockWidth_);
            AscendC::SetFlag<HardEvent::V_S>(evVS_);
            AscendC::WaitFlag<HardEvent::V_S>(evVS_);
            // 交错写回 satBuf
            LocalTensor<AccT> accView = accBigLocal_[ci * blockWidth_];
            ScatterChannel(satLocal_, accView, ci, static_cast<int32_t>(channel_), blockWidth_);
        }

        // 4) 写回整块到 sat 主区域（行偏移 r+1，列偏移 colStart+1）
        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::DataCopy(
            satGm_[(r + 1) * (width_ + 1) * channel_ + (static_cast<int64_t>(colStart_) + 1) * channel_],
            satLocal_, blockElems);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }
}

} // namespace NsIntegralImage

#endif // INTEGRAL_IMAGE_H
