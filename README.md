# IntegralImage

IntegralImage（Summed-Area Table，积分图）算子，基于 Ascend C 实现，运行于昇腾 AI Core（Ascend 910B）。

积分图使任意矩形区域内的像素和可以在 O(1) 时间内完成：

```
sat[i][j] = Σ_{0≤r<i, 0≤c<j} image[r][c]
sum(x1,y1,x2,y2) = sat[y2+1][x2+1] - sat[y1][x2+1] - sat[y2+1][x1] + sat[y1][x1]
```

## 数据格式

| 参数 | name | type | shape | dtype | format |
|---|---|---|---|---|---|
| 输入 | image | tensor | (H,W) 或 (H,W,C) | uint8, float16, float32 | ND |
| 输出 | sat | tensor | (H+1,W+1) 或 (H+1,W+1,C) | int32（整数输入）/ float32（浮点输入） | ND |

- 3D 输入按通道独立计算积分图（C 维不缩减、不跨通道累加），布局为 HWC
- 输出采用 OpenCV 惯例的物理零填充边：`sat[0][*]=sat[*][0]=0`
- dtype 映射：uint8→int32（溢出阈值约 8.4M 像素）、float16→float32、float32→float32

## 目录结构

```
├── CMakeLists.txt
├── op_graph/       # 算子原型定义（REG_OP）与 InferDataType
├── op_host/        # op_def / infershape / tiling / config
├── op_kernel/      # Ascend C kernel（HWC 交错布局，多核列分块）
└── examples/       # aclnn 功能测试（test_aclnn_integral_image.cpp）
```

## 构建与测试

```bash
# 编译 kernel
bash build.sh --pkg --soc=ascend910b --ops=integral_image -j16

# 部署
./build_out/cann-ops-cv-custom-linux.aarch64.run

# 功能测试（5 用例：uint8/fp16/fp32、单核/多核/3D）
bash build.sh --run_example integral_image eager cust --vendor_name=custom
```

## 参考算子

- OpenCV `cv::integral`: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html
- NVIDIA NPP `nppiIntegral`: https://docs.nvidia.com/cuda/archive/12.2.0/npp/group__image__integral.html
- MATLAB `integralImage`: https://www.mathworks.com/help/images/ref/integralimage.html
