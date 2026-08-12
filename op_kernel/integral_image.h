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
 * \brief IntegralImage kernel implementation (Ascend C)
 *
 * v8 design:
 *   - Input  image: (H,W) or (H,W,C), uint8/float16/float32, ND (HWC layout for 3D)
 *   - Output sat  : (H+1,W+1) or (H+1,W+1,C), int32 (integer input) / float32 (float input)
 *   - Two-phase path (2D, W >= 128):
 *       Phase A: all cores scan rows in parallel (in-row tile + scalar carry),
 *                each element is read once, results go to a GM workspace (H*W*AccT).
 *       SyncAll: hardware cross-core barrier.
 *       Phase B: each core owns one or more column tiles and performs a vector
 *                vertical accumulation, writing directly into sat with zero
 *                top row / left column boundaries.
 *   - Legacy paths: per-column block multi-core for small 2D and 3D HWC.
 */
#ifndef INTEGRAL_IMAGE_H
#define INTEGRAL_IMAGE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/math/cumsum.h"
#include "integral_image_tiling_data.h"

namespace NsIntegralImage {

using namespace AscendC;

// fp32/fp16 phase-A row scan: vector CumSum along the last axis, no last-row output
// (carry is read back from the destination tail instead).
constexpr AscendC::CumSumConfig kIntegralImageCumSumCfg = {
    true, false, false, AscendC::CumSumAlgorithm::CUMSUM_ALGORITHM_LINEBYLINE};
// Phase B processes this many rows per batch (2D copy + in-order V-pipe adds).
constexpr int32_t kPhaseBBatchRows = 8;

template <typename InT, typename AccT>
class IntegralImage {
public:
    __aicore__ inline IntegralImage() {}
    __aicore__ inline void Init(GM_ADDR image, GM_ADDR sat, GM_ADDR sqsum, GM_ADDR tilted, GM_ADDR workspace,
                                const IntegralImageTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessTilted();

private:
    __aicore__ inline void LoadBlock(LocalTensor<InT>& dst, const GlobalTensor<InT>& src, int64_t offset,
                                     int32_t elems);
    __aicore__ inline void LoadBlockAcc(LocalTensor<AccT>& dst, const GlobalTensor<AccT>& src, int64_t offset,
                                        int32_t elems);
    __aicore__ inline void LoadBlockAcc2D(LocalTensor<AccT>& dst, const GlobalTensor<AccT>& src, int64_t offset,
                                          int32_t rows, int32_t rowLen, int64_t srcPitch);
    __aicore__ inline void LoadBlock2D(LocalTensor<InT>& dst, const GlobalTensor<InT>& src, int64_t offset,
                                       int32_t rows, int32_t rowLen, int64_t srcPitch, int64_t dstRowGap);
    __aicore__ inline void StoreBlock(const GlobalTensor<AccT>& dst, const LocalTensor<AccT>& src, int64_t offset,
                                      int32_t elems);
    __aicore__ inline void StoreBlockF(const GlobalTensor<float>& dst, const LocalTensor<float>& src, int64_t offset,
                                       int32_t elems);
    __aicore__ inline void StoreBlockAcc2D(const GlobalTensor<AccT>& dst, const LocalTensor<AccT>& src,
                                           int64_t offset, int32_t rows, int32_t rowLen, int64_t dstPitch,
                                           int64_t srcRowGap);
    __aicore__ inline void StoreBlockAcc2DF(const GlobalTensor<float>& dst, const LocalTensor<float>& src,
                                            int64_t offset, int32_t rows, int32_t rowLen, int64_t dstPitch,
                                            int64_t srcRowGap);
    __aicore__ inline void LoadBlockAccF(LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t offset,
                                         int32_t elems);
    __aicore__ inline void LoadBlockAcc2DF(LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t offset,
                                           int32_t rows, int32_t rowLen, int64_t srcPitch);
    __aicore__ inline void RowScan(LocalTensor<AccT>& row, int32_t len, AccT init = static_cast<AccT>(0));
    __aicore__ inline void ExtractChannel(LocalTensor<AccT>& dst, const LocalTensor<InT>& src, int32_t c,
                                          int32_t channel, int32_t len);
    __aicore__ inline AccT SumChannel(const LocalTensor<InT>& src, int32_t c, int32_t channel, int32_t len);
    __aicore__ inline float SumChannelSq(const LocalTensor<InT>& src, int32_t c, int32_t channel, int32_t len);
    __aicore__ inline void ScatterChannel(LocalTensor<AccT>& dst, const LocalTensor<AccT>& src, int32_t c,
                                          int32_t channel, int32_t len);
    // ---- fast 2D path (C == 1): vector cast + scalar row scan ----
    __aicore__ inline void CastVec2D(const LocalTensor<AccT>& dst, const LocalTensor<InT>& src, int32_t len);
    __aicore__ inline void RowScanVec(LocalTensor<AccT>& row, int32_t len);
    __aicore__ inline void Process2D();
    // ---- two-phase path (2D, W >= 128) ----
    __aicore__ inline void ProcessTwoPhase();

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> imgBuf_;
    TBuf<TPosition::VECIN> satBuf_;
    TBuf<TPosition::VECIN> accBigBuf_;   // C*blockW AccT, per-channel column accumulators
    TBuf<TPosition::VECIN> prefixBuf_;   // C AccT, per-channel block-start compensation
    TBuf<TPosition::VECIN> zeroBigBuf_;  // 8*C AccT, zero boundary
    TBuf<TPosition::VECIN> castBuf_;
    TBuf<TPosition::VECIN> zeroBuf_;
    TBuf<TPosition::VECIN> leftBuf_;  // left block row-sum buffer (block width + zero gutter)
    TBuf<TPosition::VECIN> halfBuf_;  // u8->f16 intermediate buffer (block width + 8 halves)
    TBuf<TPosition::VECIN> wsRowBuf_; // phase B workspace row tile
    TBuf<TPosition::VECIN> lastRowBuf_; // CumSum last-row scratch (row tile width)
    TBuf<TPosition::VECIN> wsBatchBuf_; // phase B batched workspace rows
    TBuf<TPosition::VECIN> outBuf_;     // phase B batched output snapshot
    TBuf<TPosition::VECIN> imgBatchBuf_;  // phase A batched input rows (InT)
    TBuf<TPosition::VECIN> castBatchBuf_; // phase A batched fp16->fp32 cast
    TBuf<TPosition::VECIN> dstBatchBuf_;  // phase A batched CumSum output
    TBuf<TPosition::VECIN> dstSqBatchBuf_;  // phase A batched sqsum row prefix (AccT)
    TBuf<TPosition::VECIN> sqSrcBatchBuf_;  // phase A batched src^2 (fp path)
    TBuf<TPosition::VECIN> wsSqBatchBuf_;   // phase B batched sqsum workspace rows
    TBuf<TPosition::VECIN> outSqBuf_;       // phase B batched sqsum output snapshot
    TBuf<TPosition::VECIN> wsSqRowBuf_;     // phase B sqsum workspace row tile
    TBuf<TPosition::VECIN> outSqRowBuf_;    // phase B sqsum output row tile
    TBuf<TPosition::VECIN> accSqBuf_;       // phase B sqsum column accumulator
    TBuf<TPosition::VECIN> zeroSqBuf_;      // float32 zero boundary for sqsum writes
    LocalTensor<InT> imgLocal_;
    LocalTensor<AccT> satLocal_;
    LocalTensor<AccT> accBigLocal_;
    LocalTensor<AccT> prefixLocal_;
    LocalTensor<AccT> zeroBigLocal_;
    LocalTensor<AccT> castLocal_;
    LocalTensor<AccT> zeroLocal_;
    LocalTensor<AccT> leftLocal_;
    LocalTensor<half> halfLocal_;
    LocalTensor<AccT> wsRowLocal_;
    LocalTensor<AccT> lastRowLocal_;
    LocalTensor<AccT> wsBatchLocal_;
    LocalTensor<AccT> outLocal_;
    LocalTensor<InT> imgBatchLocal_;
    LocalTensor<AccT> castBatchLocal_;
    LocalTensor<AccT> dstBatchLocal_;
    // sqsum is always float32 (independent of AccT: u8->i32 sum path still uses fp32 sqsum)
    LocalTensor<float> dstSqBatchLocal_;
    LocalTensor<float> sqSrcBatchLocal_;
    LocalTensor<float> wsSqBatchLocal_;
    LocalTensor<float> outSqLocal_;
    LocalTensor<float> wsSqRowLocal_;
    LocalTensor<float> outSqRowLocal_;
    LocalTensor<float> accSqLocal_;
    LocalTensor<float> zeroSqLocal_;

    GlobalTensor<InT> imageGm_;
    GlobalTensor<AccT> satGm_;
    GlobalTensor<float> sqsumGm_;
    GlobalTensor<AccT> tiltedGm_;
    GlobalTensor<AccT> wsGm_;
    GlobalTensor<float> wsSqGm_;
    GlobalTensor<AccT> rpGm_;   // tilted row-prefix source (twoPhase: == wsGm_)
    GlobalTensor<AccT> acc1Gm_; // tilted main-diagonal prefix buffer
    GlobalTensor<AccT> acc2Gm_; // tilted anti-diagonal prefix buffer

    int64_t height_ = 0;
    int64_t width_ = 0;
    int64_t channel_ = 1;
    int32_t blockWidth_ = 0;
    int32_t colStart_ = 0;
    int64_t twoPhase_ = 0;
    int64_t activeCoreNum_ = 1;
    int64_t rowTileWidth_ = 0;
    int64_t colTileWidth_ = 0;
    int64_t colTileCount_ = 0;
    int32_t wsValid_ = 0;
    int32_t sqsumValid_ = 1;
    int32_t tiltedValid_ = 0;
    int32_t evMte2V_ = 0;
    int32_t evVMte2_ = 0;
    int32_t evMte2S_ = 0;
    int32_t evSMte2_ = 0;
    int32_t evVMte3_ = 0;
    int32_t evVS_ = 0;
    int32_t evSV_ = 0;
    int32_t evSMte3_ = 0;
    int32_t evMte3V_ = 0;
};

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::Init(GM_ADDR image, GM_ADDR sat, GM_ADDR sqsum, GM_ADDR tilted,
    GM_ADDR workspace, const IntegralImageTilingData* tilingData)
{
    height_ = tilingData->height;
    width_ = tilingData->width;
    channel_ = tilingData->channel;
    blockWidth_ = static_cast<int32_t>(tilingData->blockWidth);
    twoPhase_ = tilingData->twoPhase;
    activeCoreNum_ = tilingData->activeCoreNum;
    rowTileWidth_ = tilingData->rowTileWidth;
    colTileWidth_ = tilingData->colTileWidth;
    colTileCount_ = tilingData->colTileCount;
    colStart_ = AscendC::GetBlockIdx() * blockWidth_;

    imageGm_.SetGlobalBuffer((__gm__ InT*)image);
    satGm_.SetGlobalBuffer((__gm__ AccT*)sat);
    sqsumValid_ = (tilingData->sqsumEnabled != 0);
    if (sqsumValid_ != 0) {
        sqsumGm_.SetGlobalBuffer((__gm__ float*)sqsum);
    }
    // Optional tilted output: enabled only when the caller provides it (tiling flag).
    tiltedValid_ = (tilingData->tiltedEnabled != 0);
    if (tiltedValid_ != 0) {
        tiltedGm_.SetGlobalBuffer((__gm__ AccT*)tilted);
    }

    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);

    pipe_.InitBuffer(imgBuf_, blockElems * sizeof(InT));
    pipe_.InitBuffer(satBuf_, blockElems * sizeof(AccT));
    pipe_.InitBuffer(accBigBuf_, blockElems * sizeof(AccT));
    // UB buffer lengths aligned up to 32 bytes
    pipe_.InitBuffer(prefixBuf_, (static_cast<int32_t>(channel_) * sizeof(AccT) + 31) & ~31);
    pipe_.InitBuffer(zeroBigBuf_, (8 * static_cast<int32_t>(channel_) * sizeof(AccT) + 31) & ~31);
    pipe_.InitBuffer(castBuf_, (blockWidth_ + 8) * sizeof(AccT));
    pipe_.InitBuffer(zeroBuf_, blockWidth_ * sizeof(AccT));
    pipe_.InitBuffer(leftBuf_, (blockWidth_ + 8) * sizeof(AccT));
    pipe_.InitBuffer(halfBuf_, (blockWidth_ + 8) * sizeof(half));
    pipe_.InitBuffer(wsRowBuf_, (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    pipe_.InitBuffer(lastRowBuf_, (blockWidth_ + 8) * sizeof(AccT));
    pipe_.InitBuffer(wsBatchBuf_, kPhaseBBatchRows * (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    pipe_.InitBuffer(outBuf_, kPhaseBBatchRows * (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    // phase-A batch buffers: u8 uses imgBatch+dstBatch for batched scalar scan;
    // fp32/fp16 use imgBatch+dstBatch (+castBatch for fp16) for batched CumSum.
    // Keep VECIN small so PopStackBuffer(LCM) still has room for CumSum tmp.
    pipe_.InitBuffer(imgBatchBuf_, kPhaseBBatchRows * (blockWidth_ + 8) * sizeof(InT));
    if constexpr (!std::is_same<InT, uint8_t>::value && !std::is_same<InT, AccT>::value) {
        pipe_.InitBuffer(castBatchBuf_, kPhaseBBatchRows * (blockWidth_ + 8) * sizeof(AccT));
    }
    pipe_.InitBuffer(dstBatchBuf_, kPhaseBBatchRows * (blockWidth_ + 8) * sizeof(AccT));
    pipe_.InitBuffer(dstSqBatchBuf_, kPhaseBBatchRows * (blockWidth_ + 8) * sizeof(AccT));
    if constexpr (!std::is_same<InT, uint8_t>::value) {
        pipe_.InitBuffer(sqSrcBatchBuf_, kPhaseBBatchRows * (blockWidth_ + 8) * sizeof(AccT));
    }
    pipe_.InitBuffer(wsSqBatchBuf_, kPhaseBBatchRows * (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    pipe_.InitBuffer(outSqBuf_, kPhaseBBatchRows * (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    pipe_.InitBuffer(wsSqRowBuf_, (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    pipe_.InitBuffer(outSqRowBuf_, (static_cast<int32_t>(colTileWidth_) + 8) * sizeof(AccT));
    // shared by phase B (colTileW), Process2D (blockWidth) and 3D (blockWidth*C)
    pipe_.InitBuffer(accSqBuf_, blockElems * sizeof(AccT));
    pipe_.InitBuffer(zeroSqBuf_, (blockWidth_ + 8) * sizeof(float));
    if (workspace != nullptr) {
        // ops-cv custom op convention: the workspace GM_ADDR is the raw user workspace base.
        if (twoPhase_ == 1) {
            // [0,H*W) sum row prefixes, [H*W,2*H*W) sqsum row prefixes,
            // [2*H*W,3*H*W) tilted acc1, [3*H*W,4*H*W) tilted acc2
            wsGm_.SetGlobalBuffer((__gm__ AccT*)workspace);
            wsSqGm_.SetGlobalBuffer((__gm__ float*)workspace + height_ * width_);
            rpGm_.SetGlobalBuffer((__gm__ AccT*)workspace);
            acc1Gm_.SetGlobalBuffer((__gm__ AccT*)workspace + 2 * height_ * width_);
            acc2Gm_.SetGlobalBuffer((__gm__ AccT*)workspace + 3 * height_ * width_);
        } else {
            // legacy: [0,H*W) tilted row prefixes, [H*W,2*H*W) acc1, [2*H*W,3*H*W) acc2
            rpGm_.SetGlobalBuffer((__gm__ AccT*)workspace);
            acc1Gm_.SetGlobalBuffer((__gm__ AccT*)workspace + height_ * width_);
            acc2Gm_.SetGlobalBuffer((__gm__ AccT*)workspace + 2 * height_ * width_);
        }
        wsValid_ = 1;
    }

    imgLocal_ = imgBuf_.template Get<InT>();
    satLocal_ = satBuf_.template Get<AccT>();
    accBigLocal_ = accBigBuf_.template Get<AccT>();
    prefixLocal_ = prefixBuf_.template Get<AccT>();
    zeroBigLocal_ = zeroBigBuf_.template Get<AccT>();
    castLocal_ = castBuf_.template Get<AccT>();
    zeroLocal_ = zeroBuf_.template Get<AccT>();
    leftLocal_ = leftBuf_.template Get<AccT>();
    halfLocal_ = halfBuf_.template Get<half>();
    wsRowLocal_ = wsRowBuf_.template Get<AccT>();
    lastRowLocal_ = lastRowBuf_.template Get<AccT>();
    wsBatchLocal_ = wsBatchBuf_.template Get<AccT>();
    outLocal_ = outBuf_.template Get<AccT>();
    imgBatchLocal_ = imgBatchBuf_.template Get<InT>();
    if constexpr (!std::is_same<InT, uint8_t>::value && !std::is_same<InT, AccT>::value) {
        castBatchLocal_ = castBatchBuf_.template Get<AccT>();
    }
    dstBatchLocal_ = dstBatchBuf_.template Get<AccT>();
    dstSqBatchLocal_ = dstSqBatchBuf_.template Get<float>();
    if constexpr (!std::is_same<InT, uint8_t>::value) {
        sqSrcBatchLocal_ = sqSrcBatchBuf_.template Get<float>();
    }
    wsSqBatchLocal_ = wsSqBatchBuf_.template Get<float>();
    outSqLocal_ = outSqBuf_.template Get<float>();
    wsSqRowLocal_ = wsSqRowBuf_.template Get<float>();
    outSqRowLocal_ = outSqRowBuf_.template Get<float>();
    accSqLocal_ = accSqBuf_.template Get<float>();
    zeroSqLocal_ = zeroSqBuf_.template Get<float>();

    evMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    evVMte2_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
    evMte2S_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    evSMte2_ = GetTPipePtr()->FetchEventID(HardEvent::S_MTE2);
    evVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    evVS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    evSV_ = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    evSMte3_ = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
    evMte3V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
}

// 32B aligned blocks use DataCopy fast path, unaligned (tail / W < 32) use DataCopyPad
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
        // isPad=true: zero-pad the tail (scans need zero-padded tails)
        AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, static_cast<InT>(0)});
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

// float32 variant for the sqsum output (sqsum is always fp32, independent of AccT)
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::StoreBlockF(const GlobalTensor<float>& dst,
    const LocalTensor<float>& src, int64_t offset, int32_t elems)
{
    if (elems * static_cast<int32_t>(sizeof(float)) % 32 == 0) {
        AscendC::DataCopy(dst[offset], src, elems);
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = elems * sizeof(float);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        AscendC::DataCopyPad(dst[offset], src, p);
    }
}

// AccT workspace row tile load (phase B): 32B aligned blocks use DataCopy, tail uses DataCopyPad
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlockAcc(LocalTensor<AccT>& dst, const GlobalTensor<AccT>& src,
    int64_t offset, int32_t elems)
{
    if (elems * static_cast<int32_t>(sizeof(AccT)) % 32 == 0) {
        AscendC::DataCopy(dst, src[offset], elems);
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = elems * sizeof(AccT);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, static_cast<AccT>(0)});
    }
}

// Batched 2D workspace load (DataCopyPad, byte-unit strides): rows x rowLen AccT elements,
// consecutive rows are srcPitch elements apart in GM, zero-padded tail.
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlockAcc2D(LocalTensor<AccT>& dst, const GlobalTensor<AccT>& src,
    int64_t offset, int32_t rows, int32_t rowLen, int64_t srcPitch)
{
    DataCopyExtParams p;
    p.blockCount = rows;
    p.blockLen = rowLen * sizeof(AccT);
    p.srcStride = static_cast<uint32_t>((srcPitch - rowLen) * sizeof(AccT));
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, static_cast<AccT>(0)});
}

// Batched 2D input load (InT): rows x rowLen, GM rows srcPitch apart, UB rows dstRowGap apart
// (dstRowGap = per-row UB stride - rowLen, zero-padded tail).
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlock2D(LocalTensor<InT>& dst, const GlobalTensor<InT>& src,
    int64_t offset, int32_t rows, int32_t rowLen, int64_t srcPitch, int64_t dstRowGap)
{
    DataCopyExtParams p;
    p.blockCount = rows;
    p.blockLen = rowLen * sizeof(InT);
    p.srcStride = static_cast<uint32_t>((srcPitch - rowLen) * sizeof(InT));
    p.dstStride = static_cast<uint32_t>(dstRowGap * sizeof(InT));
    p.rsv = 0;
    AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, static_cast<InT>(0)});
}

// Batched 2D store (DataCopyPad, byte-unit strides): rows x rowLen AccT elements from a
// UB snapshot (rows srcRowGap apart) to GM rows dstPitch elements apart.
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::StoreBlockAcc2D(const GlobalTensor<AccT>& dst,
    const LocalTensor<AccT>& src, int64_t offset, int32_t rows, int32_t rowLen, int64_t dstPitch, int64_t srcRowGap)
{
    DataCopyExtParams p;
    p.blockCount = rows;
    p.blockLen = rowLen * sizeof(AccT);
    p.srcStride = static_cast<uint32_t>(srcRowGap * sizeof(AccT));
    p.dstStride = static_cast<uint32_t>((dstPitch - rowLen) * sizeof(AccT));
    p.rsv = 0;
    AscendC::DataCopyPad(dst[offset], src, p);
}

// float32 variant for the sqsum output
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::StoreBlockAcc2DF(const GlobalTensor<float>& dst,
    const LocalTensor<float>& src, int64_t offset, int32_t rows, int32_t rowLen, int64_t dstPitch, int64_t srcRowGap)
{
    DataCopyExtParams p;
    p.blockCount = rows;
    p.blockLen = rowLen * sizeof(float);
    p.srcStride = static_cast<uint32_t>(srcRowGap * sizeof(float));
    p.dstStride = static_cast<uint32_t>((dstPitch - rowLen) * sizeof(float));
    p.rsv = 0;
    AscendC::DataCopyPad(dst[offset], src, p);
}

// float32 variants for the sqsum workspace (row tile + batched 2D)
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlockAccF(LocalTensor<float>& dst,
    const GlobalTensor<float>& src, int64_t offset, int32_t elems)
{
    if (elems * static_cast<int32_t>(sizeof(float)) % 32 == 0) {
        AscendC::DataCopy(dst, src[offset], elems);
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = elems * sizeof(float);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, 0.0f});
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::LoadBlockAcc2DF(LocalTensor<float>& dst,
    const GlobalTensor<float>& src, int64_t offset, int32_t rows, int32_t rowLen, int64_t srcPitch)
{
    DataCopyExtParams p;
    p.blockCount = rows;
    p.blockLen = rowLen * sizeof(float);
    p.srcStride = static_cast<uint32_t>((srcPitch - rowLen) * sizeof(float));
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPad(dst, src[offset], p, {true, 0, 0, 0.0f});
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::RowScan(LocalTensor<AccT>& row, int32_t len, AccT init)
{
    AccT running = init;
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
        // aicore 不允许 float <-> uint8 直接 cast，统一经 int32 中转
        dst.SetValue(i, static_cast<AccT>(static_cast<int32_t>(src.GetValue(c + i * channel))));
    }
}

template <typename InT, typename AccT>
__aicore__ inline AccT IntegralImage<InT, AccT>::SumChannel(const LocalTensor<InT>& src, int32_t c, int32_t channel,
    int32_t len)
{
    AccT sum = static_cast<AccT>(0);
    for (int32_t i = 0; i < len; i++) {
        sum += static_cast<AccT>(static_cast<int32_t>(src.GetValue(c + i * channel)));
    }
    return sum;
}

template <typename InT, typename AccT>
__aicore__ inline float IntegralImage<InT, AccT>::SumChannelSq(const LocalTensor<InT>& src, int32_t c,
    int32_t channel, int32_t len)
{
    float sum = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        const AccT v = static_cast<AccT>(static_cast<int32_t>(src.GetValue(c + i * channel)));
        sum += static_cast<float>(v) * static_cast<float>(v);
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

// Cast: fp32 avoids copy (caller uses input buffer directly); u8 scalar Cast; fp16 vector Cast.
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::CastVec2D(const LocalTensor<AccT>& dst, const LocalTensor<InT>& src,
    int32_t len)
{
    if constexpr (std::is_same<InT, AccT>::value) {
        // fp32 -> fp32: no conversion needed (caller already uses the input buffer)
        return;
    } else if constexpr (std::is_same<InT, uint8_t>::value) {
        // u8 -> s32: scalar Cast (count/mask-array u8 vector Cast is a known HW defect on this platform)
        for (int32_t i = 0; i < len; i++) {
            dst.SetValue(i, static_cast<AccT>(static_cast<int32_t>(src.GetValue(i))));
        }
    } else {
        // fp16 -> fp32
        int32_t rem = len;
        int32_t off = 0;
        while (rem > 0) {
            int32_t cnt = (rem > 64) ? 64 : rem;
            AscendC::Cast<float, half>(dst[off], src[off], RoundMode::CAST_NONE, static_cast<uint32_t>(cnt));
            off += cnt;
            rem -= cnt;
        }
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::RowScanVec(LocalTensor<AccT>& row, int32_t len)
{
    RowScan(row, len);
}

// ================= two-phase path (2D, C == 1, W >= 128) =================
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::ProcessTwoPhase()
{
    if (wsValid_ == 0) {
        return;
    }
    const int64_t blockIdx = AscendC::GetBlockIdx();
    const int64_t rowTileW = rowTileWidth_;
    const int64_t colTileW = colTileWidth_;
    const int64_t activeCore = activeCoreNum_;

    // ---------- Phase A: horizontal scan ----------
    if constexpr (std::is_same<InT, uint8_t>::value) {
        // u8: contiguous row bands, 8-row batches, fused scalar cast+scan per row
        // (GetValue/Add/SetValue per element instead of separate cast then scan)
        const int64_t rowsPerCore = (height_ + activeCore - 1) / activeCore;
        const int64_t rowStart = blockIdx * rowsPerCore;
        const int64_t rowEnd = (height_ < rowStart + rowsPerCore) ? height_ : (rowStart + rowsPerCore);
        for (int64_t rb = rowStart; rb < rowEnd; rb += kPhaseBBatchRows) {
            const int32_t rows = static_cast<int32_t>(
                ((rowEnd - rb) < kPhaseBBatchRows) ? (rowEnd - rb) : kPhaseBBatchRows);
            AccT carryVals[kPhaseBBatchRows];
            float carrySqVals[kPhaseBBatchRows];
            for (int32_t k = 0; k < rows; k++) {
                carryVals[k] = static_cast<AccT>(0);
                carrySqVals[k] = 0.0f;
            }
            for (int64_t col = 0; col < width_; col += rowTileW) {
                const int32_t tileW = static_cast<int32_t>((rowTileW < width_ - col) ? rowTileW : (width_ - col));
                const bool aligned = (tileW * static_cast<int32_t>(sizeof(InT)) % 32) == 0;
                if (aligned) {
                    // 32B-aligned tile: 8-row batched fused scalar cast+scan
                    AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                    LoadBlock2D(imgBatchLocal_, imageGm_, rb * width_ + col, rows, tileW, width_, 0);
                    AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
                    AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);
                    if (sqsumValid_ != 0) {
                        for (int32_t k = 0; k < rows; k++) {
                            AccT running = carryVals[k];
                            float runningSq = carrySqVals[k];
                            for (int32_t i = 0; i < tileW; i++) {
                                const AccT v =
                                    static_cast<AccT>(static_cast<int32_t>(imgBatchLocal_.GetValue(k * tileW + i)));
                                running += v;
                                runningSq += static_cast<float>(v) * static_cast<float>(v);
                                dstBatchLocal_.SetValue(k * tileW + i, running);
                                dstSqBatchLocal_.SetValue(k * tileW + i, runningSq);
                            }
                            carryVals[k] = running;
                            carrySqVals[k] = runningSq;
                        }
                    } else {
                        for (int32_t k = 0; k < rows; k++) {
                            AccT running = carryVals[k];
                            for (int32_t i = 0; i < tileW; i++) {
                                const AccT v =
                                    static_cast<AccT>(static_cast<int32_t>(imgBatchLocal_.GetValue(k * tileW + i)));
                                running += v;
                                dstBatchLocal_.SetValue(k * tileW + i, running);
                            }
                            carryVals[k] = running;
                        }
                    }
                    AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
                    AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
                    StoreBlockAcc2D(wsGm_, dstBatchLocal_, rb * width_ + col, rows, tileW, width_, 0);
                    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                    if (sqsumValid_ != 0) {
                        // dstSqBatch is written by the scalar pipe -> S_MTE3, not V_MTE3
                        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
                        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
                        StoreBlockAcc2DF(wsSqGm_, dstSqBatchLocal_, rb * width_ + col, rows, tileW, width_, 0);
                        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                    }
                } else {
                    // unaligned tail tile: per-row fallback (proven single-row path)
                    for (int32_t k = 0; k < rows; k++) {
                        const int64_t row = rb + k;
                        // DataCopyPad requires 32B-aligned UB source: use an aligned row stride
                        const int32_t rowStride = ((tileW + 7) / 8) * 8;
                        AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                        AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                        LoadBlock(imgLocal_, imageGm_, row * width_ + col, tileW);
                        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
                        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);
                        if (sqsumValid_ != 0) {
                            AccT running = carryVals[k];
                            float runningSq = carrySqVals[k];
                            for (int32_t i = 0; i < tileW; i++) {
                                const AccT v = static_cast<AccT>(static_cast<int32_t>(imgLocal_.GetValue(i)));
                                running += v;
                                runningSq += static_cast<float>(v) * static_cast<float>(v);
                                castLocal_.SetValue(i, running);
                                dstSqBatchLocal_.SetValue(k * rowStride + i, runningSq);
                            }
                            carryVals[k] = running;
                            carrySqVals[k] = runningSq;
                        } else {
                            AccT running = carryVals[k];
                            for (int32_t i = 0; i < tileW; i++) {
                                const AccT v = static_cast<AccT>(static_cast<int32_t>(imgLocal_.GetValue(i)));
                                running += v;
                                castLocal_.SetValue(i, running);
                            }
                            carryVals[k] = running;
                        }
                        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
                        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
                        StoreBlock(wsGm_, castLocal_, row * width_ + col, tileW);
                        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                        if (sqsumValid_ != 0) {
                            // dstSqBatch is written by the scalar pipe -> S_MTE3, not V_MTE3
                            AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
                            AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
                            StoreBlockF(wsSqGm_, dstSqBatchLocal_[k * rowStride], row * width_ + col, tileW);
                            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                        }
                    }
                }
            }
        }
    } else {
        // fp32/fp16: contiguous row bands, 8-row batches, vector CumSum per column tile
        const int64_t rowsPerCore = (height_ + activeCore - 1) / activeCore;
        const int64_t rowStart = blockIdx * rowsPerCore;
        const int64_t rowEnd = (height_ < rowStart + rowsPerCore) ? height_ : (rowStart + rowsPerCore);
        for (int64_t rb = rowStart; rb < rowEnd; rb += kPhaseBBatchRows) {
            const int32_t rows = static_cast<int32_t>(
                ((rowEnd - rb) < kPhaseBBatchRows) ? (rowEnd - rb) : kPhaseBBatchRows);
            AccT carryVals[kPhaseBBatchRows];
            float carrySqVals[kPhaseBBatchRows];
            for (int32_t k = 0; k < rows; k++) {
                carryVals[k] = static_cast<AccT>(0);
                carrySqVals[k] = 0.0f;
            }
            for (int64_t col = 0; col < width_; col += rowTileW) {
                const int32_t tileW = static_cast<int32_t>((rowTileW < width_ - col) ? rowTileW : (width_ - col));
                const bool aligned = (tileW * static_cast<int32_t>(sizeof(InT)) % 32) == 0;
                if (aligned) {
                    // 32B-aligned tile: 8-row batched vector CumSum
                    AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                    LoadBlock2D(imgBatchLocal_, imageGm_, rb * width_ + col, rows, tileW, width_, 0);
                    AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);

                    LocalTensor<AccT> srcBatch = castBatchLocal_;
                    if constexpr (std::is_same<InT, AccT>::value) {
                        srcBatch = imgBatchLocal_;
                    } else {
                        // fp16 -> fp32 vector cast, one row per call (aligned count)
                        for (int32_t k = 0; k < rows; k++) {
                            AscendC::Cast<AccT, InT>(castBatchLocal_[k * tileW], imgBatchLocal_[k * tileW],
                                                     RoundMode::CAST_NONE, static_cast<uint32_t>(tileW));
                        }
                    }
                    if (sqsumValid_ != 0) {
                        // sqsum: src^2 -> CumSum (same batch, vector square)
                        AscendC::Mul(sqSrcBatchLocal_, srcBatch, srcBatch, rows * tileW);
                        AscendC::CumSum<AccT, kIntegralImageCumSumCfg>(dstSqBatchLocal_, lastRowLocal_,
                            sqSrcBatchLocal_, AscendC::CumSumInfo{static_cast<uint32_t>(rows),
                                                                  static_cast<uint32_t>(tileW)});
                    }
                    AscendC::CumSum<AccT, kIntegralImageCumSumCfg>(dstBatchLocal_, lastRowLocal_, srcBatch,
                        AscendC::CumSumInfo{static_cast<uint32_t>(rows), static_cast<uint32_t>(tileW)});
                    if (col > 0) {
                        for (int32_t k = 0; k < rows; k++) {
                            AscendC::Adds(dstBatchLocal_[k * tileW], dstBatchLocal_[k * tileW], carryVals[k], tileW);
                            if (sqsumValid_ != 0) {
                                AscendC::Adds(dstSqBatchLocal_[k * tileW], dstSqBatchLocal_[k * tileW],
                                              carrySqVals[k], tileW);
                            }
                        }
                    }
                    AscendC::SetFlag<HardEvent::V_S>(evVS_);
                    AscendC::WaitFlag<HardEvent::V_S>(evVS_);
                    for (int32_t k = 0; k < rows; k++) {
                        carryVals[k] = dstBatchLocal_.GetValue(k * tileW + tileW - 1);
                        if (sqsumValid_ != 0) {
                            carrySqVals[k] = dstSqBatchLocal_.GetValue(k * tileW + tileW - 1);
                        }
                    }
                    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                    StoreBlockAcc2D(wsGm_, dstBatchLocal_, rb * width_ + col, rows, tileW, width_, 0);
                    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                    if (sqsumValid_ != 0) {
                        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                        StoreBlockAcc2DF(wsSqGm_, dstSqBatchLocal_, rb * width_ + col, rows, tileW, width_, 0);
                        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                    }
                } else {
                    // unaligned tail tile: per-row vector CumSum fallback
                    const int32_t innerAligned = ((tileW + 7) / 8) * 8; // float 32B alignment
                    for (int32_t k = 0; k < rows; k++) {
                        const int64_t row = rb + k;
                        AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                        AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                        LoadBlock(imgLocal_, imageGm_, row * width_ + col, tileW);
                        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
                        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);
                        LocalTensor<AccT> src = castLocal_;
                        if constexpr (std::is_same<InT, AccT>::value) {
                            src = imgLocal_;
                        } else {
                            CastVec2D(castLocal_, imgLocal_, tileW);
                            AscendC::SetFlag<HardEvent::V_S>(evVS_);
                            AscendC::WaitFlag<HardEvent::V_S>(evVS_);
                        }
                        if (sqsumValid_ != 0) {
                            // sqsum row prefix: src^2 -> CumSum
                            LocalTensor<AccT> sqSrcRow = sqSrcBatchLocal_[k * tileW];
                            LocalTensor<AccT> dstSqRow = dstSqBatchLocal_[k * tileW];
                            AscendC::Mul(sqSrcRow, src, src, tileW);
                            AscendC::CumSum<AccT, kIntegralImageCumSumCfg>(dstSqRow, lastRowLocal_, sqSrcRow,
                                AscendC::CumSumInfo{1, static_cast<uint32_t>(innerAligned)});
                        }
                        AscendC::CumSum<AccT, kIntegralImageCumSumCfg>(leftLocal_, lastRowLocal_, src,
                            AscendC::CumSumInfo{1, static_cast<uint32_t>(innerAligned)});
                        if (col > 0) {
                            AscendC::Adds(leftLocal_, leftLocal_, carryVals[k], tileW);
                            if (sqsumValid_ != 0) {
                                AscendC::Adds(dstSqBatchLocal_[k * tileW], dstSqBatchLocal_[k * tileW],
                                              carrySqVals[k], tileW);
                            }
                        }
                        AscendC::SetFlag<HardEvent::V_S>(evVS_);
                        AscendC::WaitFlag<HardEvent::V_S>(evVS_);
                        carryVals[k] = leftLocal_.GetValue(tileW - 1);
                        if (sqsumValid_ != 0) {
                            carrySqVals[k] = dstSqBatchLocal_.GetValue(k * tileW + tileW - 1);
                        }
                        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                        StoreBlock(wsGm_, leftLocal_, row * width_ + col, tileW);
                        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                        if (sqsumValid_ != 0) {
                            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                            StoreBlockF(wsSqGm_, dstSqBatchLocal_[k * tileW], row * width_ + col, tileW);
                            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                        }
                    }
                }
            }
        }
    }

    // ---------- cross-core barrier: all launched cores must reach this point ----------
    AscendC::SyncAll();

    // ---------- Phase B: per-column-tile vector vertical accumulation ----------
    for (int64_t tile = blockIdx; tile < colTileCount_; tile += activeCore) {
        const int64_t colStart = tile * colTileW;
        const int32_t tileW = static_cast<int32_t>((colTileW < width_ - colStart) ? colTileW : (width_ - colStart));

        // zero top row: sat[0][colStart .. colStart+tileW], tile 0 also covers sat[0][0]
        AscendC::Duplicate(satLocal_, static_cast<AccT>(0), static_cast<int32_t>(colTileW + 8));
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        if (tile == 0) {
            StoreBlock(satGm_, satLocal_, 0, tileW + 1);
        } else {
            StoreBlock(satGm_, satLocal_, colStart + 1, tileW);
        }
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);

        if (sqsumValid_ != 0) {
            // sqsum zero top row: sqsum[0][colStart .. colStart+tileW] (tile 0 also covers sqsum[0][0])
            AscendC::Duplicate(zeroSqLocal_, 0.0f, static_cast<int32_t>(colTileW + 8));
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            if (tile == 0) {
                StoreBlockF(sqsumGm_, zeroSqLocal_, 0, tileW + 1);
            } else {
                StoreBlockF(sqsumGm_, zeroSqLocal_, colStart + 1, tileW);
            }
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }

        AscendC::Duplicate(accBigLocal_, static_cast<AccT>(0), static_cast<int32_t>(colTileW));
        if (sqsumValid_ != 0) {
            AscendC::Duplicate(accSqLocal_, 0.0f, static_cast<int32_t>(colTileW));
        }
        if (tileW % 8 == 0) {
            // ---- batched path: 8 rows per iteration, sync overhead divided by 8 ----
            AscendC::Duplicate(zeroBigLocal_, static_cast<AccT>(0), 8);
            for (int64_t rb = 0; rb < height_; rb += kPhaseBBatchRows) {
                const int32_t rows = static_cast<int32_t>(
                    ((height_ - rb) < kPhaseBBatchRows) ? (height_ - rb) : kPhaseBBatchRows);
                AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                LoadBlockAcc2D(wsBatchLocal_, wsGm_, rb * width_ + colStart, rows, tileW, width_);
                if (sqsumValid_ != 0) {
                    LoadBlockAcc2DF(wsSqBatchLocal_, wsSqGm_, rb * width_ + colStart, rows, tileW, width_);
                }
                AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
                AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
                if (tile == 0) {
                    for (int32_t k = 0; k < rows; k++) {
                        AscendC::Add(accBigLocal_, accBigLocal_, wsBatchLocal_[k * tileW], tileW);
                        AscendC::Adds(outLocal_[k * tileW], accBigLocal_, static_cast<AccT>(0), tileW);
                        if (sqsumValid_ != 0) {
                            AscendC::Add(accSqLocal_, accSqLocal_, wsSqBatchLocal_[k * tileW], tileW);
                            AscendC::Adds(outSqLocal_[k * tileW], accSqLocal_, 0.0f, tileW);
                        }
                    }
                    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                    // left column zeros (per-row small stores) + main row segments (batched)
                    for (int32_t k = 0; k < rows; k++) {
                        StoreBlock(satGm_, zeroBigLocal_, (rb + k + 1) * (width_ + 1), 1);
                        if (sqsumValid_ != 0) {
                            StoreBlockF(sqsumGm_, zeroSqLocal_, (rb + k + 1) * (width_ + 1), 1);
                        }
                    }
                    StoreBlockAcc2D(satGm_, outLocal_, (rb + 1) * (width_ + 1) + 1, rows, tileW, width_ + 1, 0);
                    if (sqsumValid_ != 0) {
                        StoreBlockAcc2DF(sqsumGm_, outSqLocal_, (rb + 1) * (width_ + 1) + 1, rows, tileW,
                                         width_ + 1, 0);
                    }
                } else {
                    for (int32_t k = 0; k < rows; k++) {
                        AscendC::Add(accBigLocal_, accBigLocal_, wsBatchLocal_[k * tileW], tileW);
                        AscendC::Adds(outLocal_[k * tileW], accBigLocal_, static_cast<AccT>(0), tileW);
                        if (sqsumValid_ != 0) {
                            AscendC::Add(accSqLocal_, accSqLocal_, wsSqBatchLocal_[k * tileW], tileW);
                            AscendC::Adds(outSqLocal_[k * tileW], accSqLocal_, 0.0f, tileW);
                        }
                    }
                    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                    StoreBlockAcc2D(satGm_, outLocal_, (rb + 1) * (width_ + 1) + colStart + 1, rows, tileW,
                                    width_ + 1, 0);
                    if (sqsumValid_ != 0) {
                        StoreBlockAcc2DF(sqsumGm_, outSqLocal_, (rb + 1) * (width_ + 1) + colStart + 1, rows, tileW,
                                         width_ + 1, 0);
                    }
                }
                AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            }
        } else {
            // ---- unaligned tail tile: per-row fallback ----
            AscendC::Duplicate(zeroBigLocal_, static_cast<AccT>(0), 8);
            for (int64_t r = 0; r < height_; r++) {
                AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
                AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                LoadBlockAcc(wsRowLocal_, wsGm_, r * width_ + colStart, tileW);
                if (sqsumValid_ != 0) {
                    LoadBlockAccF(wsSqRowLocal_, wsSqGm_, r * width_ + colStart, tileW);
                }
                AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
                AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
                AscendC::Add(accBigLocal_, accBigLocal_, wsRowLocal_, tileW);
                if (sqsumValid_ != 0) {
                    AscendC::Add(accSqLocal_, accSqLocal_, wsSqRowLocal_, tileW);
                }
                AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                if (tile == 0) {
                    // left column zero + main segment
                    StoreBlock(satGm_, zeroBigLocal_, (r + 1) * (width_ + 1), 1);
                    if (sqsumValid_ != 0) {
                        StoreBlockF(sqsumGm_, zeroSqLocal_, (r + 1) * (width_ + 1), 1);
                    }
                    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
                    StoreBlock(satGm_, accBigLocal_, (r + 1) * (width_ + 1) + 1, tileW);
                    if (sqsumValid_ != 0) {
                        StoreBlockF(sqsumGm_, accSqLocal_, (r + 1) * (width_ + 1) + 1, tileW);
                    }
                } else {
                    StoreBlock(satGm_, accBigLocal_, (r + 1) * (width_ + 1) + colStart + 1, tileW);
                    if (sqsumValid_ != 0) {
                        StoreBlockF(sqsumGm_, accSqLocal_, (r + 1) * (width_ + 1) + colStart + 1, tileW);
                    }
                }
                AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            }
        }
    }
}

// 2D fast path: whole contiguous block, no Extract/Scatter, no prefix re-read of left blocks.
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::Process2D()
{
    const int32_t segW = (width_ - colStart_ < blockWidth_) ? static_cast<int32_t>(width_ - colStart_) : blockWidth_;

    // zero boundary buffer init (V write, MTE3 read)
    AscendC::Duplicate(zeroBigLocal_, static_cast<AccT>(0), 8 * static_cast<int32_t>(channel_));
    AscendC::Duplicate(accBigLocal_, static_cast<AccT>(0), blockWidth_);
    if (sqsumValid_ != 0) {
        AscendC::Duplicate(accSqLocal_, 0.0f, blockWidth_);
        AscendC::Duplicate(zeroSqLocal_, 0.0f, blockWidth_);
    }
    // row 0 zero fill: sat[0][colStart+1 .. colStart+segW]
    AscendC::Duplicate(satLocal_, static_cast<AccT>(0), blockWidth_);
    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    StoreBlock(satGm_, satLocal_, (int64_t)colStart_ + 1, segW);
    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    if (sqsumValid_ != 0) {
        // row 0 zero fill: sqsum[0][colStart+1 .. colStart+segW]
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlockF(sqsumGm_, zeroSqLocal_, (int64_t)colStart_ + 1, segW);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }
    if (colStart_ == 0) {
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlock(satGm_, zeroBigLocal_, 0, 1);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        if (sqsumValid_ != 0) {
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlockF(sqsumGm_, zeroSqLocal_, 0, 1);
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }

    for (int64_t r = 0; r < height_; r++) {
        // previous round's V reads of imgLocal_ must finish before re-covering
        AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
        AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);

        // column 0 zero fill (core 0): sat[r+1][0] = 0
        if (colStart_ == 0) {
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlock(satGm_, zeroBigLocal_, (r + 1) * (width_ + 1), 1);
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            if (sqsumValid_ != 0) {
                AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                StoreBlockF(sqsumGm_, zeroSqLocal_, (r + 1) * (width_ + 1), 1);
                AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            }
        }

        // block-start compensation: sum of all left blocks' rows (vector add + scalar tail)
        AccT ps = static_cast<AccT>(0);
        float psSq = 0.0f;
        for (int64_t off = 0; off < colStart_; off += blockWidth_) {
            AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
            AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
            LoadBlock(imgLocal_, imageGm_, r * width_ + off, blockWidth_);
            AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
            AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);
            LocalTensor<AccT> workLeft = leftLocal_;
            if constexpr (std::is_same<InT, AccT>::value) {
                workLeft = imgLocal_;
            }
            CastVec2D(workLeft, imgLocal_, blockWidth_);
            AscendC::SetFlag<HardEvent::V_S>(evVS_);
            AscendC::WaitFlag<HardEvent::V_S>(evVS_);
            for (int32_t i = 0; i < blockWidth_; i++) {
                const AccT v = workLeft.GetValue(i);
                ps += v;
                if (sqsumValid_ != 0) {
                    psSq += static_cast<float>(v) * static_cast<float>(v);
                }
            }
        }

        // main block load
        AscendC::SetFlag<HardEvent::V_MTE2>(evVMte2_);
        AscendC::WaitFlag<HardEvent::V_MTE2>(evVMte2_);
        LoadBlock(imgLocal_, imageGm_, r * width_ + colStart_, segW);
        AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);

        LocalTensor<AccT> work = castLocal_;
        if constexpr (std::is_same<InT, AccT>::value) {
            work = imgLocal_;
        }
        CastVec2D(work, imgLocal_, segW);
        if (sqsumValid_ != 0) {
            // sqsum row prefix: scalar square accumulation into dstSqBatchLocal_[0] (float32)
            {
                LocalTensor<float> sqWork = dstSqBatchLocal_[0];
                float runningSq = 0.0f;
                for (int32_t i = 0; i < segW; i++) {
                    const AccT v = work.GetValue(i);
                    runningSq += static_cast<float>(v) * static_cast<float>(v);
                    sqWork.SetValue(i, runningSq);
                }
                if (psSq != 0.0f) {
                    for (int32_t i = 0; i < segW; i++) {
                        sqWork.SetValue(i, sqWork.GetValue(i) + psSq);
                    }
                }
                AscendC::SetFlag<HardEvent::S_V>(evSV_);
                AscendC::WaitFlag<HardEvent::S_V>(evSV_);
                AscendC::Add(accSqLocal_, accSqLocal_, sqWork, segW);
            }
        }
        RowScanVec(work, segW);
        if (ps != static_cast<AccT>(0)) {
            AscendC::SetFlag<HardEvent::S_V>(evSV_);
            AscendC::WaitFlag<HardEvent::S_V>(evSV_);
            AscendC::Duplicate(zeroLocal_, ps, segW);
            AscendC::Add(work, work, zeroLocal_, segW);
        }
        // column accumulation: sat[r+1][*] = sat[r][*] + row prefix
        AscendC::Add(accBigLocal_, accBigLocal_, work, segW);

        // write back sat[r+1][colStart+1 .. colStart+segW]
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlock(satGm_, accBigLocal_, (r + 1) * (width_ + 1) + colStart_ + 1, segW);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        if (sqsumValid_ != 0) {
            // write back sqsum[r+1][colStart+1 .. colStart+segW]
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlockF(sqsumGm_, accSqLocal_, (r + 1) * (width_ + 1) + colStart_ + 1, segW);
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }
}

// Tilted integral (OpenCV integral3), multi-core diagonal-prefix decomposition:
//   tilted[y][x] = acc1[y][x] - acc2[y][x], where
//   acc1[y][x] = sum_{k<=y} rowPrefix[k][x+(y-k)]   (main-diagonal prefix)
//   acc2[y][x] = sum_{k<=y} rowPrefix[k][x-(y-k)-1] (anti-diagonal prefix)
// Main and anti diagonals are fully independent -> 40 cores split by diagonal,
// no cross-core communication, two SyncAll barriers total.
// Boundaries (defined): tilted[y][W] = 0, tilted[H][*] = 0, first row comes out
// automatically as src[0][*] (matches OpenCV), first column is the closed-form value.
template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::ProcessTilted()
{
    // Optional output: when the caller does not provide a tilted tensor, skip entirely
    // (all cores see the same flag, so no SyncAll deadlock).
    if (tiltedValid_ == 0) {
        return;
    }
    const int64_t blockIdx = AscendC::GetBlockIdx();
    const int64_t coreNum = AscendC::GetBlockNum();
    const int64_t plane = height_ * width_;
    const int64_t OHW = width_ + 1;
    for (int64_t c = 0; c < channel_; c++) {
        const int64_t srcOff = c * plane;
        const int64_t dstOff = c * (height_ + 1) * OHW;

        // T0: row prefixes (legacy path only; two-phase reuses wsGm_/rpGm_)
        if (twoPhase_ != 1) {
            for (int64_t row = blockIdx; row < height_; row += coreNum) {
                AccT run = static_cast<AccT>(0);
                for (int64_t x = 0; x < width_; x++) {
                    run += static_cast<AccT>(static_cast<int32_t>(imageGm_.GetValue(srcOff + row * width_ + x)));
                    rpGm_.SetValue(row * width_ + x, run);
                }
            }
        }
        AscendC::SyncAll();

        // T1: main-diagonal prefixes acc1 (diagonal u = x+y)
        for (int64_t u = blockIdx; u < height_ + width_ - 1; u += coreNum) {
            AccT acc = static_cast<AccT>(0);
            const int64_t kmax = (u < height_) ? u : (height_ - 1);
            for (int64_t k = 0; k <= kmax; k++) {
                const int64_t t = u - k;
                if (t >= 0) {
                    acc += (t >= width_) ? rpGm_.GetValue(k * width_ + width_ - 1)
                                         : rpGm_.GetValue(k * width_ + t);
                }
                if (t >= 0 && t < width_) {
                    acc1Gm_.SetValue(k * width_ + t, acc);
                }
            }
        }

        // T2: anti-diagonal prefixes acc2 (diagonal v = x-y)
        for (int64_t v = blockIdx - (height_ - 1); v <= width_ - 1; v += coreNum) {
            AccT acc = static_cast<AccT>(0);
            for (int64_t y = 0; y < height_; y++) {
                const int64_t t = v + y - 1;
                if (t >= 0) {
                    acc += (t >= width_) ? rpGm_.GetValue(y * width_ + width_ - 1)
                                         : rpGm_.GetValue(y * width_ + t);
                }
                const int64_t x = y + v;
                if (x >= 0 && x < width_) {
                    acc2Gm_.SetValue(y * width_ + x, acc);
                }
            }
        }
        AscendC::SyncAll();

        // T3: merge tilted[row][x] = acc1 - acc2 (row=0..H-1, x=0..W-1).
        // Row 0 automatically equals src[0][*] (matches OpenCV). Column W and row H are
        // defined as zero boundaries (OpenCV leaves them undefined).
        for (int64_t row = blockIdx; row < height_; row += coreNum) {
            for (int64_t x = 0; x < width_; x++) {
                const int64_t idx = row * width_ + x;
                tiltedGm_.SetValue(dstOff + row * OHW + x, acc1Gm_.GetValue(idx) - acc2Gm_.GetValue(idx));
            }
        }
        if (blockIdx == 0) {
            // right column (x = W) zero for rows 0..H-1, and bottom row (y = H) zero
            for (int64_t y = 0; y < height_; y++) {
                tiltedGm_.SetValue(dstOff + y * OHW + width_, static_cast<AccT>(0));
            }
            for (int64_t x = 0; x < OHW; x++) {
                tiltedGm_.SetValue(dstOff + height_ * OHW + x, static_cast<AccT>(0));
            }
        }
        AscendC::SyncAll();
    }
}

template <typename InT, typename AccT>
__aicore__ inline void IntegralImage<InT, AccT>::Process()
{
    // two-phase path: all launched cores must reach SyncAll (no early return here)
    if (twoPhase_ == 1) {
        ProcessTwoPhase();
        ProcessTilted();
        return;
    }
    // legacy path: extra cores with colStart beyond width exit early
    if (colStart_ >= width_) {
        ProcessTilted();
        return;
    }
    // 2D (C == 1) uses the vector fast path; 3D keeps the scalar HWC path
    if (channel_ == 1) {
        Process2D();
        ProcessTilted();
        return;
    }
    const int32_t blockElems = blockWidth_ * static_cast<int32_t>(channel_);
    const int32_t segW = (width_ - colStart_ < blockWidth_)
                             ? static_cast<int32_t>(width_ - colStart_)
                             : blockWidth_;
    const int32_t segElems = segW * static_cast<int32_t>(channel_);

    AscendC::Duplicate(zeroLocal_, static_cast<AccT>(0), blockWidth_);
    AscendC::Duplicate(accBigLocal_, static_cast<AccT>(0), blockElems);
    if (sqsumValid_ != 0) {
        AscendC::Duplicate(accSqLocal_, 0.0f, blockElems);
    }
    AscendC::Duplicate(zeroBigLocal_, static_cast<AccT>(0), 8 * static_cast<int32_t>(channel_));

    // row 0 zero fill: each core writes its own column block sat[0][colStart+1 .. colStart+segW]
    AscendC::Duplicate(satLocal_, static_cast<AccT>(0), blockElems);
    AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
    AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    StoreBlock(satGm_, satLocal_, (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
    AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
    AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    if (sqsumValid_ != 0) {
        // row 0 zero fill: sqsum[0][colStart+1 .. colStart+segW] (all channels)
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlockF(sqsumGm_, accSqLocal_, (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }
    // core 0 writes sat[0][0..C-1] = 0
    if (colStart_ == 0) {
        AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        StoreBlock(satGm_, zeroBigLocal_, 0, static_cast<int32_t>(channel_));
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        if (sqsumValid_ != 0) {
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlockF(sqsumGm_, zeroSqLocal_, 0, static_cast<int32_t>(channel_));
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }

    for (int64_t r = 0; r < height_; r++) {
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);

        // 1) column 0 zero fill (core 0): sat[r+1][0..C-1] = 0
        if (colStart_ == 0) {
            AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
            StoreBlock(satGm_, zeroBigLocal_, (r + 1) * (width_ + 1) * channel_, static_cast<int32_t>(channel_));
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            if (sqsumValid_ != 0) {
                AscendC::SetFlag<HardEvent::V_MTE3>(evVMte3_);
                AscendC::WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                StoreBlockF(sqsumGm_, zeroSqLocal_, (r + 1) * (width_ + 1) * channel_,
                           static_cast<int32_t>(channel_));
                AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
                AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
            }
        }

        // 2) block-start compensation: sum + sqsum of all left blocks, per channel independently
        AscendC::Duplicate(prefixLocal_, static_cast<AccT>(0), static_cast<int32_t>(channel_));
        if (sqsumValid_ != 0) {
            AscendC::Duplicate(outSqRowLocal_, 0.0f, static_cast<int32_t>(channel_));
        }
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
                if (sqsumValid_ != 0) {
                    outSqRowLocal_.SetValue(static_cast<int32_t>(c),
                        outSqRowLocal_.GetValue(static_cast<int32_t>(c)) +
                            SumChannelSq(imgLocal_, static_cast<int32_t>(c), static_cast<int32_t>(channel_),
                                         blockWidth_));
                }
            }
        }

        // 3) load main block (this core's segW columns)
        AscendC::SetFlag<HardEvent::S_MTE2>(evSMte2_);
        AscendC::WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        LoadBlock(imgLocal_, imageGm_, r * width_ * channel_ + static_cast<int64_t>(colStart_) * channel_, segElems);
        AscendC::SetFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        AscendC::SetFlag<HardEvent::MTE2_S>(evMte2S_);
        AscendC::WaitFlag<HardEvent::MTE2_S>(evMte2S_);

        // 4) per channel: extract -> row scan -> +prefixStart -> column propagate -> interleave into satBuf
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

        // 5) write back the whole block to sat main region
        AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
        AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        StoreBlock(satGm_, satLocal_,
            (r + 1) * (width_ + 1) * channel_ + (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
        AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);

        if (sqsumValid_ != 0) {
            // 6) sqsum per channel: extract -> scalar square-prefix -> +psSq -> column propagate -> interleave
            for (int64_t c = 0; c < channel_; c++) {
                const int32_t ci = static_cast<int32_t>(c);
                ExtractChannel(castLocal_, imgLocal_, ci, static_cast<int32_t>(channel_), segW);
                LocalTensor<float> sqWork = dstSqBatchLocal_[0];
                float runningSq = 0.0f;
                for (int32_t i = 0; i < segW; i++) {
                    const AccT v = static_cast<AccT>(static_cast<int32_t>(castLocal_.GetValue(i)));
                    runningSq += static_cast<float>(v) * static_cast<float>(v);
                    sqWork.SetValue(i, runningSq);
                }
                const float psq = outSqRowLocal_.GetValue(ci);
                if (psq != 0.0f) {
                    for (int32_t i = 0; i < segW; i++) {
                        sqWork.SetValue(i, sqWork.GetValue(i) + psq);
                    }
                }
                AscendC::SetFlag<HardEvent::S_V>(evSV_);
                AscendC::WaitFlag<HardEvent::S_V>(evSV_);
                AscendC::Add(accSqLocal_[ci * blockWidth_], accSqLocal_[ci * blockWidth_], sqWork, segW);
                AscendC::SetFlag<HardEvent::V_S>(evVS_);
                AscendC::WaitFlag<HardEvent::V_S>(evVS_);
                // interleave accSq channel into the float output snapshot (sqsum is fp32)
                for (int32_t i = 0; i < segW; i++) {
                    dstSqBatchLocal_.SetValue(ci + i * static_cast<int32_t>(channel_),
                        accSqLocal_.GetValue(ci * blockWidth_ + i));
                }
            }
        }

        if (sqsumValid_ != 0) {
            // 7) write back the whole block to sqsum main region
            AscendC::SetFlag<HardEvent::S_MTE3>(evSMte3_);
            AscendC::WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            StoreBlockF(sqsumGm_, dstSqBatchLocal_,
                (r + 1) * (width_ + 1) * channel_ + (static_cast<int64_t>(colStart_) + 1) * channel_, segElems);
            AscendC::SetFlag<HardEvent::MTE3_V>(evMte3V_);
            AscendC::WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }
    ProcessTilted();
}

} // namespace NsIntegralImage

#endif // INTEGRAL_IMAGE_H
