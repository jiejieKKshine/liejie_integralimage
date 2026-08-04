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
 * v5 设计（HWC 布局 + 非对齐宽度支持）：
 *   - 输入 image: (H,W) 或 (H,W,C)，uint8/float16/float32，ND
 *   - 输出 sat  : (H+1,W+1) 或 (H+1,W+1,C)，int32（整数输入）/ float32（浮点输入）
 *   - HWC 下每通道平面跨步存储，每行整块加载 blockW*C 连续元素，
 *     UB 内按通道提取（步长 C）处理，再交错写回。
 *   - 非对齐宽度：每核处理 segW = min(blockW, W-colStart) 列；
 *     32B 对齐的块用 DataCopy，非对齐尾部（含 W<32）用 DataCopyPad。
 *   - 每通道独立列累加器，多核按列分块 + 块起始补偿。
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
    __aicore__ inline void LoadBlock(LocalTensor<InT>& dst, const GlobalTensor<InT>& src, int64_t offset,
                                     int32_t elems);
    __aicore__ inline void StoreBlock(const GlobalTensor<AccT>& dst, const LocalTensor<AccT>& src, int64_t offset,
                                      int32_t elems);
    __aicore__ inline void RowScan(LocalTensor<AccT>& row, int32_t len);
    __aicore__ inline void ExtractChannel(LocalTensor<AccT>& dst, const LocalTensor<InT>& src, int32_t c,
                                          int32_t channel, int32_t len);
    __aicore__ inline AccT SumChannel(const LocalTensor<InT>& src, int32_t c, int32_t channel, int32_t len);
    __aicore__ inline void ScatterChannel(LocalTensor<AccT>& dst, const LocalTensor<AccT>& src, int32_t c,
                                          int32_t channel, int32_t len);

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> imgBuf_;
    TBuf<TPosition::VECIN> satBuf_;
    TBuf<TPosition::VECIN> accBigBuf_;   // C*blockW AccT，每通道列累加器
    TBuf<TPosition::VECIN> prefixBuf_;   // C AccT，每通道块起始补偿
    TBuf<TPosition::VECIN> zeroBigBuf_;  // 8*C AccT，零填充边界
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
    int32_t evVMte3_ = 0;
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
    // 保护：核数上限下的多余核直接退出（colStart 超出宽度时不处理）
    if (colStart_ >= width_) {
        return;
    }

    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);

    imageGm_.SetGlobalBuffer((__gm__ InT*)image);
    satGm_.SetGlobalBuffer((__gm__ AccT*)sat);

    pipe_.InitBuffer(imgBuf_, blockElems * sizeof(InT));
    pipe_.InitBuffer(satBuf_, blockElems * sizeof(AccT));
    pipe_.InitBuffer(accBigBuf_, blockElems * sizeof(AccT));
    // UB Buffer 长度统一向上对齐到 32 字节
    pipe_.InitBuffer(prefixBuf_, (static_cast<int32_t>(channel_) * sizeof(AccT) + 31) & ~31);
    pipe_.InitBuffer(zeroBigBuf_, (8 * static_cast<int32_t>(channel_) * sizeof(AccT) + 31) & ~31);
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
    evVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    evVS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    evSV_ = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    evSMte3_ = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
    evMte3V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
}

// 32B 对齐块走 DataCopy 快路径，非对齐（尾部/W<32）走 DataCopyPad
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlock(LocalTensor<InT>& dst, const GlobalTensor<InT>& src,
    int64_t offset, int32_t elems)
{
    if (elems * static_cast<int32_t>(sizeof(InT)) % 32 == 0) {
        AscendC::DataCopy(dst, src[offset], elems);
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = elems * sizeof(InT);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        AscendC::DataCopyPad(dst, src[offset], p, {false, 0, 0, static_cast<InT>(0)});
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::StoreBlock(const GlobalTensor<AccT>& dst, const LocalTensor<AccT>& src,
    int64_t offset, int32_t elems)
{
    if (elems * static_cast<int32_t>(sizeof(AccT)) % 32 == 0) {
        AscendC::DataCopy(dst[offset], src, elems);
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = elems * sizeof(AccT);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        AscendC::DataCopyPad(dst[offset], src, p);
    }
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
    // 多余核（colStart 超出宽度）不参与计算
    if (colStart_ >= width_) {
        return;
    }
    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);
    // 本核实际处理列数（最后一个核可能 < blockWidth，非对齐）
    const int32_t segW = (width_ - colStart_ < blockWidth_)
                             ? static_cast<int32_t>(width_ - colStart_)
                             : blockWidth_;
    const int32_t segElems = segW * static_cast<int32_t>(channel_);

    AscendC::Duplicate(zeroLocal_, static_cast<AccT>(0), blockWidth_);
    AscendC::Duplicate(accBigLocal_, static_cast<AccT>(0), blockElems);
    // 零边界缓冲初始化（V 管线写，后续 MTE3 读）
    AscendC::Duplicate(zeroBigLocal_, static_cast<AccT>(0), 8 * static_cast<int32_t>(channel_));

    // 第 0 行零填充：每核写自己列块 sat[0][colStart+1 .. colStart+segW]（偏移 +C）
    AscendC::Duplicate(satLocal_, static_cast<AccT>(0), blockElems);
    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    StoreBlock(satGm_, satLocal_, (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    // core 0 补 sat[0][0..C-1] = 0（C 个元素，StoreBlock 处理非 32B）
    if (colStart_ == 0) {
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlock(satGm_, zeroBigLocal_, 0, static_cast<int32_t>(channel_));
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }

    for (int64_t r = 0; r < height_; r++) {
        // 上一轮 S 标量读 imgLocal_ 完成后方可覆盖
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);

        // 1) 第 0 列零填充（core 0）：写 sat[r+1][0..C-1] = 0（C 个元素）
        if (colStart_ == 0) {
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlock(satGm_, zeroBigLocal_, (r + 1) * (width_ + 1) * channel_, static_cast<int32_t>(channel_));
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }

        // 2) 块起始补偿：左侧所有块（完整 blockW 宽）的列和，每通道独立
        AscendC::Duplicate(prefixLocal_, static_cast<AccT>(0), static_cast<int32_t>(channel_));
        AscendC::SetFlag<HardEvent::V_S>(evVS_);
        AscendC::WaitFlag<HardEvent::V_S>(evVS_);
        for (int64_t off = 0; off < colStart_; off += blockWidth_) {
            AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
            AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            LoadBlock(imgLocal_, imageGm_, r * width_ * channel_ + off * channel_, blockElems);
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

        // 3) 加载主块（本核 segW 列）——最后加载，不被左侧块覆盖
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        LoadBlock(imgLocal_, imageGm_, r * width_ * channel_ + static_cast<int64_t>(colStart_) * channel_, segElems);
        AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);

        // 4) 每通道：提取 -> 行前缀和 -> +prefixStart -> 列传播 -> 交错写 satBuf
        for (int64_t c = 0; c < channel_; c++) {
            const int32_t ci = static_cast<int32_t>(c);
            ExtractChannel(castLocal_, imgLocal_, ci, static_cast<int32_t>(channel_), segW);
            RowScan(castLocal_, segW);
            const AccT ps = prefixLocal_.GetValue(ci);
            for (int32_t i = 0; i < segW; i++) {
                castLocal_.SetValue(i, castLocal_.GetValue(i) + ps);
            }
            AscendC::SetFlag<HardEvent::S_V>(evSV_);
            AscendC::WaitFlag<HardEvent::S_V>(evSV_);
            AscendC::Add(accBigLocal_[ci * blockWidth_], accBigLocal_[ci * blockWidth_], castLocal_, segW);
            AscendC::SetFlag<HardEvent::V_S>(evVS_);
            AscendC::WaitFlag<HardEvent::V_S>(evVS_);
            LocalTensor<AccT> accView = accBigLocal_[ci * blockWidth_];
            ScatterChannel(satLocal_, accView, ci, static_cast<int32_t>(channel_), segW);
        }

        // 5) 写回整块到 sat 主区域（行偏移 r+1，列偏移 colStart+1）
        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        StoreBlock(satGm_, satLocal_,
            (r + 1) * (width_ + 1) * channel_ + (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }
}

} // namespace NsIntegralImage

#endif // INTEGRAL_IMAGE_H
