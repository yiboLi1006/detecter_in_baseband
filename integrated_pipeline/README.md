# Integrated Pipeline v8.1.2 —— 相干消色散与脉冲检测流水线

## 概述

本流水线整合 **VDIF/Mark5B 数据读取 → 消色散（非相干或相干）→ 脉冲检测 → CSV/PSRFITS/图像输出** 的全流程，适用于脉冲星单脉冲搜索。

## v8.1.2 绘图更新

将已确认的脉冲显示方案用于流水线自动出图：

- Raw 瀑布图按实际 DM、参考频率及通道频率覆盖完整色散扫频，两端各留 1 ms；按接近 16 μs 的整数采样数平均，仅用于显示。
- Dedispersed 瀑布图保留原采样，在拟合中心约 ±25 μs 展示。两个面板使用独立的线性色标，范围外的值映射为端点颜色；原数据不截值。
- Raw 色标采用离色散轨迹超过 0.4 ms 的背景 median+0.5σ 至 median+3σ；Dedispersed 采用中心 ±1 ms 邻域、排除 ±100 μs 后的背景 median 至 median+6σ。σ 为 1.4826×MAD。全 NaN、恒定背景及短边界窗口有有限色标回退。
- 两个瀑布图各有色条，名称为 Raw / Dedispersed，不再画遮挡脉冲的中心/边界引导线。**光变曲线的取数、窗口、Gaussian 拟合、阈值、标线和原布局不变**。
- 绘图入口可接收 `dm` / `ref_freq`（MHz），缺省从消色散 HDU 的 `DM` / `REFFREQ` 读取，参考频率再回退 `OBSFREQ`；不把 Raw 的 `DM=0` 当作应用值。磁盘和内存 FITS 使用同一 DATA-column TDIM 解析路径，绘图读取不修改 HDU 数组。
- 输出仍为 300 dpi PNG，使用无损优化。已验证的8个论文片段图片约414–548 kB；其他数据若超过1 MB会提示，保留分辨率和颜色精度。

版本标识为 `v8.1.2`，启动信息及自动生成的输出目录、CSV 文件名随之更新。INI 的整数 `[output] version` 含义不变。此次仅修改绘图显示，不改变检测、候选筛选、TOA、CSV 数值或 PSRFITS 数据。

## v8.1.1 修复

修复单子带 VDIF 读取为一维电压数组时，`flag_xband_data()` 将样本数误当子带数、反复输出 `mask length != subband number` 并跳过 mask 的问题。单子带 mask 现在保留电压 dtype：`mask_sband = 1` 保留数据，`0` 屏蔽有限电压样本；二维多子带仍使用原有处理路径。BV 的 `withsubband = False` 保持不变，无需调整 ini。

版本标识更新为 `v8.1.1`，启动信息及自动生成的输出目录、CSV 文件名随之更新；ini 的 `[output] version`（PSRFITS 文件名中的版本号）含义不变。

## v8.1 说明

v8.1 基于 v7.13（`51cadbe`），新增两项可选功能。通道化、逐 chunk 归一化、RFI 去除、检测、拟合、TOA 公式、CSV 列、PSRFITS 格式、基带片段和多进程划分均不变：

- `[dm_correction] method = coherent`：在通道化之前，对每个实采样子带做电压域相干消色散，消除通道内色散展宽，见下文“相干消色散”。在显式 DM 和其他参数相同、`max_fwhm_ms` 留空时，留空或填 `incoherent` 保持 v7.13 的科学计算行为；回归测试中同一数据上 CSV 逐列相同。
- `[detection] max_fwhm_ms`：拟合 FWHM 上限（ms），用于剔除逐 chunk 归一化台阶等造成的 ms 级宽假候选；留空不限制。
- `[observation] dm_source` 的缺省回退值改为 56.73，即 P26219/L26219（2026-08-07）实测的最佳 DM。

本版与旧分支 `feature/v8.1-directory-mode`（Mark5B 目录模式）无关，不包含目录输入。

## v7.13 约定（沿用）

本版从 `v7.12`（`cc8bdb0`）重新实现，不继承旧 v7.13 的 TOA 换算，也不包含 v8 目录输入。保留原有消色散、检测、高斯拟合和 SNR 筛选算法，仅增加严格处理上限、最终小时／分钟耗时及两组 TOA 输出。

- `max_files` 必须填写：只有字面值 `False` 表示无上限；非负整数表示 HDUList 全局编号上限（不含），`0` 不处理。空值、缺项、`None`、`True`、`false`、负数和小数均报错。
- `max_files` 不是“检出脉冲后保存的文件数”，也不是原始数据文件数量。`start_file=2, max_files=5` 处理编号 `[2, 5)`，单／多进程一致；`False` 处理到输入末尾。
- 最终耗时示例：`### USED time: 2 h 07.5 min`；进度行保持原样。
- INI 中频率均使用 **MHz**。`ref_freq` 留空时沿用 `center_freq`，显式填写时采用该参考频率，不再换算回中心频率。

## 文件说明

| 文件 | 功能 |
|------|------|
| `integrated_pipeline.py` | **主入口**，读取配置文件，协调各模块完成完整流水线 |
| `dm_correction_module.py` | 非相干消色散模块，提供 `dm_correct_hdulist()` 原地修改 PSRFITS HDUList |
| `coherent_dedisp_module.py` | 相干消色散模块（v8.1），提供 `plan_coherent()`、chirp 构造和 `CoherentDedispReader` 读取器包装 |
| `pulse_detection_module.py` | 脉冲检测模块，提供 `detect_pulses_in_hdulist()` 从 HDUList 检测脉冲 |
| `vdif_segment_writer.py` | 基带数据段保存模块，按帧对齐复制 VDIF/Mark5B 字节范围 |
| `pulse_plotter.py` | 脉冲瀑布图绘制模块，生成 raw vs DM-corrected 对比图 |
| `pipeline.ini` | 配置文件模板 |
| `tests/` | 回归测试：`test_v713.py`、`test_v81_coherent.py`、`test_subband_mask.py`（v8.1.1 单子带与多子带兼容性）、`test_pulse_plotter.py`（v8.1.2 显示与读取兼容性） |

## 依赖

```
numpy, scipy, pandas, astropy, baseband, matplotlib, psutil
```

## 快速开始

在安装依赖的 Python 环境中，先修改 `pipeline.ini` 的输入路径和观测参数，再在本子目录执行：

```bash
python integrated_pipeline.py pipeline.ini
```

这是正式数据运行命令；回归测试不需要观测文件，见文末。

## 配置文件说明（pipeline.ini）

配置文件采用 INI 格式，分为以下节（section）：

### [paths]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vdif_file` | str | — | 输入 VDIF/Mark5B 文件路径 |
| `data_format` | str | `vdif` | 数据格式：`vdif` 或 `mark5b` |

### [observation]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_rate` | str | `1024*u.MHz` | 采样率（支持 `*u.MHz` 表达式） |
| `telescope` | str | `Badary` | 望远镜名称 |
| `source_name` | str | `B0531+21` | 脉冲星名称 |
| `dm_source` | float | `56.73` | 脉冲星的 DM 值 (pc/cm³)；示例模板填写 56.72，应按实际观测修改（默认值 56.73 仅在未填写时生效） |
| `t_start` | float | `0.0` | 从文件的起始时间偏移（秒） |
| `ref_time` | str | — | Mark5B 格式必需，参考时间，格式如 `2025-12-23T17:05:00.0` |
| `start_file` | int | `0` | 断点续跑时的起始文件编号 |

### [output]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_subints_per_file` | int | `60` | 每个 PSRFITS 文件包含的 subint 数量 |
| `max_files` | 非负整数 / `False` | 必填；示例为 `99999` | HDUList 全局编号上限（不含），`False` 不设上限，`0` 不处理；禁止空值 |
| `version` | int | `0` | 输出文件版本号，写入文件名 |

### [frequency]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `center_freq` | float | `2419.8` | 所有子带的总中心频率 (MHz) |
| `subbands` | str | `0` | 子带索引列表，逗号分隔，如 `8,9,10,11,12,13,14,15` |
| `USB` | str | `L` | 边带：`U`（上边带）或 `L`（下边带），支持逗号分隔的列表 |
| `mask_sband` | str | `1` | 子带掩码，0=屏蔽, 1=保留，长度需与 subbands 一致 |
| `subband_width` | float | `512` | 每个子带带宽 (MHz) |
| `withsubband` | bool | `False` | 是否启用于子带读取 |
| `nchan` | int | `1` | 通道数（供 baseband 读取器使用） |
| `subband_centers` | str | `2419.8` | 各子带中心频率，逗号分隔。长度需与 subbands 一致时启用连续频率轴拼接 |

### [processing]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nchans` | int | `512` | 每个子带的频率通道数（FFT 点数） |
| `chunk_size` | int | `1048576` | 每次读取的采样点数（每个 subint 的样本数） |
| `reduction_factor` | int | `1024` | 时域降采样因子 |
| `calib_bandpass` | bool | `False` | 是否在子带内进行 B 样条带通校准 |
| `flag_band_edge` | int | `0` | 每个子带边缘需标记的通道数 |

**参数一致性约束**：需满足 `nchans == reduction_factor / 2`，程序启动时自动检查。

### [dm_correction]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | str/空 | 空（= `incoherent`） | 消色散方式：`incoherent` 逐通道平移（v7.13）；`coherent` 通道化前电压域相干消色散。其他值报错 |
| `ref_freq` | float/空 | 空 | DM 校正参考频率 (MHz)；留空则使用 `center_freq`。两种方式都以它为零延迟参考 |

### [detection]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `amp_snr_threshold` | float | `4.0` | 初始峰值检测的幅度 SNR 阈值 |
| `flux_snr_threshold` | float | `3.5` | 高斯拟合后最终脉冲筛选的流量 SNR 阈值 |
| `Fit-quality_SNR` | float | `3.0` | 拟合质量 SNR（flux / flux_err）阈值，拒绝拟合不佳的候选 |
| `max_fwhm_ms` | float/空 | 空 | 拟合 FWHM 上限（ms，v8.1），须为有限正数；留空不限制。Crab 建议 `1` |
| `peak_distance` | int | `5000` | 峰值之间最小距离（采样点数），用于去除脉冲肩部/伪影 |
| `sigma_remove_rfi_frequency` | float | `5.0` | 频域 RFI 去除的 sigma 值 |
| `sigma_remove_rfi_time_frequency` | float | `9.0` | 时频域 RFI 去除的 sigma 值 |

### [detection_manual_mask]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `freq_mask_ranges` | str | 空 | 手动频率掩码范围 (MHz)，每行一对 `start,end`，支持多行。留空禁用 |

示例：
```
freq_mask_ranges = 0,2233
                   2510,9999
                   2404.4,2475
```

### [integrated_output]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | str | `./output` | 总输出目录；自动生成带程序版本号和时间戳的 CSV、原始 PSRFITS／基带片段、图片路径 |

### [performance]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cleanup_every_n_hdulists` | int | `50` | 每 N 个 hdulist 后回收 baseband 读取器（释放内部缓存）。设为 0 禁用 |
| `n_processes` | int | `1` | `1` 为单进程，较大值启用多进程；只为上限内的非空区间创建 worker |

## 相干消色散（v8.1）

### 原理与实现

- **非相干**（v7.13）：先通道化为强度，再按各通道中心频率的延迟整体平移。通道内的色散展宽 Δt ≈ 2·K·DM·Δν/ν³ 无法消除：S 波段 v1（Δν = 0.25 MHz）约 10 μs，L 波段 v1 约 40 μs。
- **相干**：在通道化之前，用色散传递函数的逆（chirp）乘每个实采样子带的电压，时间分辨率只受子带带宽限制。
- **实现**（`coherent_dedisp_module.py`）：
  1. 每个 hdulist 第一次被读取时，读入 `[h0 − 历史, h1 + 前瞻]` 的原始电压。历史和前瞻即参考频率与最高、最低带边之间的色散延迟，另加 2% + 2048 个样本的余量。
  2. 逐子带做 overlap-save：`rfft → × chirp → irfft`。chirp 相位用 float64 计算，零延迟参考取 `ref_freq`（留空即 `center_freq`），LSB 子带取共轭。
  3. 裁剪到 `[h0, h1)` 后缓存，按原 chunk 边界返回，时间戳仍取自原读取器。之后的通道化、归一化、检测、拟合与非相干相同，只是内存检测 HDUList 不再平移通道。

### 使用要求

- 实采样，且 `sample_rate = 2 × subband_width`；多子带须填写与 `subbands` 等长的 `subband_centers`。不满足时启动即报错。
- 启动时打印一行：`Dedisp : coherent ... overlap -历史/+前瞻 ms  FFT=...  extra≈... GiB/process`；额外内存乘以进程数超过物理内存 70% 时给出警告。

### 典型配置（DM 56.73）

| 配置 | 历史 / 前瞻 (ms) | FFT 长度 | 每 hdulist FFT 块数 | 额外内存估计 (GiB/进程) |
|------|------|------|------|------|
| HP/JL S v1（`p26219_HP`） | 2.49 / 2.71 | 4 194 304 | 9 | 1.24 |
| HP/JL S v2（`p26219_HP_v2`） | 2.49 / 2.71 | 4 194 304 | 14 | 1.88 |
| UR/KM S v1（`p26219_ur_*`） | 2.07 / 2.22 | 2 097 152 | 35 | 3.35 |
| UR S v2（`p26219_ur_*_v2`） | 2.07 / 2.22 | 2 097 152 | 27 | 2.61 |
| HP/UR L v1（`l26219_*`） | 9.93 / 11.38 | 69 120 000 | 1 | 3.83 |
| HP/UR L v2（`l26219_*_v2`） | 9.93 / 11.38 | 54 000 000 | 1 | 2.99 |
| Badary（`pipeline.ini`） | 7.32 / 10.07 | 81 920 000 | 1 | 1.76 |

实测峰值内存（单进程、含绘图）：HP S v1 2.57 → 3.41 GiB，UR S v1 7.95 → 10.33 GiB，HP L v1 3.56 → 5.85 GiB。
实测耗时（与生产相同的并行方式）：
- HP S v1：100 个 hdulist、10 进程，6:00 → 7:20（+22%）；
- UR S v1：30 个 hdulist、5 进程，11:29 → 8:26（−27%，UR 的非相干通道平移本身很慢）；
- HP L v1：2 个 hdulist、单进程，1:21 → 1:55（+42%）。

### 已知局限（使用前必读）

1. **不可分辨的脉冲会被 Fit-quality 判据拒绝**。相干后的脉冲可能窄于 1 个 bin，此时高斯宽度约束不住，Fit-quality SNR（flux / flux_err）趋近 0。v7.13 生产结果的 109 个 HP 脉冲（v1，4 μs）在相干模式下的情况：
   - 只重新检出 81 个（74%）；
   - 19 个仅因 Fit-quality 被拒，拟合 σ ≤ 0.57 bin，其中不乏强脉冲（SNR_fit 10–74）；
   - 7 个在相干模式下没有候选。它们的生产 FWHM 只有 5–6 μs，小于 v1 非相干约 10 μs 的仪器展宽，很可能是生产目录中的噪声假检出；
   - 2 个 SNR_fit 处于阈值边缘。

   改用 v2（1 μs）后能重新检出 91 个（84%），强脉冲都能找回；漏掉的 18 个都是弱脉冲（生产 SNR ≤ 7.8），原因是 bin 变细后宽脉冲每个 bin 的 S/N 下降，而检测只做单 bin 阈值。**在拟合判据针对不可分辨脉冲修正之前，相干模式不建议直接替代 v1 的生产运行。**
2. **chunk 边界台阶**：逐 chunk 通量归一化在 chunk 边界产生的电平台阶，非相干时被通道平移抹平，相干后变陡，会形成宽假候选。实测中这些候选紧贴 chunk 边界（0–5 个 bin 以内），v1 下约每 100 个 hdulist 出现 1–2 个，FWHM 0.22–3.6 ms。`max_fwhm_ms = 1` 只能剔除较宽的一部分。
3. **DM 精度**：相干收益对 DM 敏感，而且 DM 误差会进入站间时差（group0006：DM 从 56.79 改为 56.73，相干 HP−UR 移动 −1.3 μs）。各站必须使用同一 DM，现有配置统一用 56.72（P26219/L26219 2026-08-07 实测最佳范围 56.72–56.74）。
4. **TOA 约定**：
   - 相干模式不受非相干“频率标签高半个通道”的影响，同一数据上 TOA 比非相干早 K·DM·Δν/ν³（S v1 约 5 μs，S v2 / L v1 约 20 μs）。新旧结果混用时要区分。
   - 时间戳取 FFT 块起点、TOA 偏早 tbin/2 的约定两种方式相同；各站 tbin 一致时，这一项在站间差中抵消。
5. **文件首尾与读取失败**：文件开头和结尾各约一个历史/前瞻宽度的数据只能部分消色散。原始电压读取失败时，凡是与坏段相距在前瞻/历史范围内的 chunk 都会被跳过，并打印 `Error processing chunk`。

### 实测验证（2026-10-11，NTSC）

| 检查 | 结果 |
|------|------|
| 缺省（非相干）与 v7.13，group0006 HP/UR segment | CSV 全部列逐字一致 |
| group0006 共同脉冲，v1，DM 56.79 | SNR_det：HP 18.5 → 24.1，UR 25.9 → 32.3；FWHM 12 → 9 μs |
| 同上，DM 56.73 | SNR_det：HP 19.3 → 31.0，UR 27.4 → 37.3；FWHM：HP 7.2 μs，UR 8.4 μs |
| HP−UR（TOA_Inf），DM 56.79 | 非相干 220826.23 μs，相干 220826.16 μs |
| 单进程 vs 双 worker（真实数据，4 个 hdulist） | 物理量完全一致，TOA 差 0 ns |
| 生产 109 个 HP 脉冲重检 | 见上文“已知局限”第 1 条 |

## 输出文件

每次检测到脉冲时，会生成以下文件：

```
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}.fits          # 原始 PSRFITS（仅含检测到脉冲的 hdulist）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_segment.vdif   # 对应的基带数据段（如启用）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_pulse{序号}.png # 脉冲瀑布图（如启用 plot_output_dir）
```

全部扫描结束后，有检出脉冲才在 `output_path` 下生成带版本号和时间戳的 CSV 文件，包含所有检出脉冲的精确到达时间、幅度、FWHM、SNR 等信息。DM 校正仅用于内存检测，不额外保存消色散后的 PSRFITS；保存的原始 PSRFITS 仍未消色散（两种消色散方式相同，头部 `DM = 0`）。相干模式下，只有检出脉冲时才按该 hdulist 的 chunk 重读原始电压，生成原始 PSRFITS 和瀑布图的 Raw 面板；瀑布图的 DM Corrected 面板显示相干消色散后的数据。

### 两组 TOA 字段

| 字段 | 含义 |
|------|------|
| `TOA_Ref_Freq_MJD` / `TOA_Ref_Freq_UTC` | 实际参考频率下的拟合 TOA；保留 v7.12 原有时间结果 |
| `TOA_Inf_Freq_MJD` / `TOA_Inf_Freq_UTC` | 从参考频率 TOA 扣除其相对无穷大频率的色散延迟 |

例如：`center_freq=1400`、`ref_freq` 为空时输出 1400 MHz 与无穷大频率 TOA；`center_freq=1400`、`ref_freq=1200` 时输出 **1200 MHz** 与无穷大频率 TOA。

换算公式（DM 单位 pc/cm³，频率 MHz，延迟秒）：

```text
delay_sec = 4.1488064239e3 * DM / ref_freq**2
TOA_inf = TOA_ref - delay_sec
```

实际 `DM` 和 `REFFREQ` 从已完成消色散的内存 HDUList 读取，与检测数据一致。DM 必须是有限非负数，频率必须是有限正数；`DM=0` 时两组 TOA 相同。此换算不是太阳系质心校正，两组时间均保持站上 UTC。

旧 `Precise_JD1`、`Precise_JD2`、`Precise_Abs_MJD_Str` 等字段保持原有含义。新增 MJD 列在写出前使用双分量 `Time` 转高精度十进制字符串（小数点后 15 位），UTC 保留 9 位小数；这只是数值表示精度，不代表测量精度。读取 CSV 时如需保留完整时间精度，不要先转成 float64 MJD：

```python
import pandas as pd
from astropy.time import Time

df = pd.read_csv(csv_path, dtype={
    'TOA_Ref_Freq_MJD': str,
    'TOA_Inf_Freq_MJD': str,
})
t_ref = Time(df['TOA_Ref_Freq_MJD'].to_numpy(), format='mjd', scale='utc')
t_inf = Time(df['TOA_Inf_Freq_MJD'].to_numpy(), format='mjd', scale='utc')
```

## 回归测试

`tests/test_v713.py` 包含严格配置、任务边界、耗时、合成 HDUList 的 TOA／CSV 精度，以及约 12 KB 合成 VDIF 的单进程／双 worker CLI 测试。

`tests/test_v81_coherent.py`（v8.1）包含：
- 相干核心：只由色散延迟定律构造的扫频，在 USB/LSB 下压缩到 ±1 样本；反边带对照；块边界无关；带边延迟与手算值一致；掩码子带原样返回；读到文件末尾不越界；坏输入只影响相交的 chunk；时间戳取自原读取器。
- 配置解析：`method`、`max_fwhm_ms`、相干模式的启动校验。
- 脉宽上限筛选。
- 约 290 KB 合成色散 VDIF 的 CLI 端到端测试：相干 TOA 复现注入时刻；非相干相对相干偏晚的量等于半通道偏差 K·DM·Δν/ν³；两种方式写出的原始 PSRFITS 逐字节相同；相干模式单进程与双 worker 结果一致。

临时文件放入 `_tmp/` 并自动清理，不访问观测数据。

`tests/test_subband_mask.py`（v8.1.1）覆盖单子带浮点电压的保留、屏蔽与 NaN mask，空数组、单样本及非连续视图，二维多子带兼容性、mask 长度不匹配的原有回退，以及 BV 参数下通道化输出的逐字节一致性。

安装依赖后，在 `integrated_pipeline/` 目录运行：

```bash
python -B -m unittest discover -s tests -v
```

2026-09-15 v7.13 已在 NTSC `pulsar`（Python 3.10.19、astropy 6.1.7、baseband 4.3.0）通过 22 项测试。同一合成脉冲在 1400 / 1200 MHz 参考频率下，与 v7.12 的原有 21 个检测字段逐项一致，仅新增 4 个 TOA 字段；同一份合成色散数据采用两种参考频率后，无穷大频率 TOA 在测试容差内一致。

2026-10-11 v8.1 在同一环境通过全部 48 项测试（v7.13 的 22 项原样通过，另有 v8.1 的 26 项）。真实数据验证见“相干消色散”一节。服务器测试均在独立临时目录执行并已清理，没有覆盖正式流水线。

2026-10-14 v8.1.1 在 NTSC 通过全部 56 项测试（原有 48 项及新增 8 项），全部 29 份服务器 ini 解析通过。发布前的 BV、HP、JL 实数据共 15 组对照覆盖原始读取、相干消色散和原始重读，修复前后电压及通道化数组的 dtype 和字节一致；单子带合成脉冲的三种运行方式中，CSV 全字段及原始 PSRFITS 数据、时间偏移也与修复前一致。单子带误报消失，mask 屏蔽生效；未完整重跑全部观测文件。

2026-10-14 v8.1.2 在 NTSC 通过66项回归测试（原56项和新增10项绘图测试），29份服务器INI均解析通过。真实L26219 HP片段的50块重放中，完整图像与已确认论文绘图方案逐字节一致，检测CSV逐字节一致，原始PSRFITS表数据逐字节一致（生成日期和校验和变化）。未重跑完整观测。
