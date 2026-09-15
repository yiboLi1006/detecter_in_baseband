# Integrated Pipeline v7.13 —— 基于 v7.12 重建的脉冲检测流水线

## 概述

本流水线整合 **VDIF/Mark5B 数据读取 → 消色散（DM Correction）→ 脉冲检测 → CSV/PSRFITS/图像输出** 的全流程，适用于脉冲星单脉冲搜索。

## v7.13 重建说明

本版从 `v7.12`（`cc8bdb0`）重新实现，不继承旧 v7.13 的 TOA 换算，也不包含 v8 目录输入。保留原有消色散、检测、高斯拟合和 SNR 筛选算法，仅增加严格处理上限、最终小时／分钟耗时及两组 TOA 输出。

- `max_files` 必须填写：只有字面值 `False` 表示无上限；非负整数表示 HDUList 全局编号上限（不含），`0` 不处理。空值、缺项、`None`、`True`、`false`、负数和小数均报错。
- `max_files` 不是“检出脉冲后保存的文件数”，也不是原始数据文件数量。`start_file=2, max_files=5` 处理编号 `[2, 5)`，单／多进程一致；`False` 处理到输入末尾。
- 最终耗时示例：`### USED time: 2 h 07.5 min`；进度行保持原样。
- INI 中频率均使用 **MHz**。`ref_freq` 留空时沿用 `center_freq`，显式填写时采用该参考频率，不再换算回中心频率。

## 文件说明

| 文件 | 功能 |
|------|------|
| `integrated_pipeline.py` | **主入口**，读取配置文件，协调各模块完成完整流水线 |
| `dm_correction_module.py` | DM 消色散模块，提供 `dm_correct_hdulist()` 原地修改 PSRFITS HDUList |
| `pulse_detection_module.py` | 脉冲检测模块，提供 `detect_pulses_in_hdulist()` 从 HDUList 检测脉冲 |
| `vdif_segment_writer.py` | 基带数据段保存模块，按帧对齐复制 VDIF/Mark5B 字节范围 |
| `pulse_plotter.py` | 脉冲瀑布图绘制模块，生成 raw vs DM-corrected 对比图 |
| `pipeline.ini` | 配置文件模板 |

## 依赖

```
numpy, scipy, pandas, astropy, baseband, matplotlib, psutil
```

## 快速开始

NTSC 服务器上使用已有 `pulsar` 环境，在本子目录执行：

```bash
/home/lyb/anaconda3/envs/pulsar/bin/python integrated_pipeline.py pipeline.ini
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
| `dm_source` | float | `56.79` | 脉冲星的 DM 值 (pc/cm³) |
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
| `ref_freq` | float/空 | 空 | DM 校正参考频率 (MHz)；留空则使用 `center_freq` |

### [detection]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `amp_snr_threshold` | float | `4.0` | 初始峰值检测的幅度 SNR 阈值 |
| `flux_snr_threshold` | float | `3.5` | 高斯拟合后最终脉冲筛选的流量 SNR 阈值 |
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

## 输出文件

每次检测到脉冲时，会生成以下文件：

```
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}.fits          # 原始 PSRFITS（仅含检测到脉冲的 hdulist）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_segment.vdif   # 对应的基带数据段（如启用）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_pulse{序号}.png # 脉冲瀑布图（如启用 plot_output_dir）
```

全部扫描结束后，有检出脉冲才在 `output_path` 下生成带版本号和时间戳的 CSV 文件，包含所有检出脉冲的精确到达时间、幅度、FWHM、SNR 等信息。DM 校正仅用于内存检测，不额外保存消色散后的 PSRFITS；保存的原始 PSRFITS 仍未消色散。

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

`tests/test_v713.py` 包含严格配置、任务边界、耗时、合成 HDUList 的 TOA／CSV 精度，以及约 12 KB 合成 VDIF 的单进程／双 worker CLI 测试。临时文件放入 `_tmp/` 并自动清理，不访问观测数据。

在 NTSC 的独立测试副本根目录运行：

```bash
/home/lyb/anaconda3/envs/pulsar/bin/python -B -m unittest discover -s tests -v
```

2026-09-15 已在 NTSC `pulsar`（Python 3.10.19、astropy 6.1.7、baseband 4.3.0）通过 22 项测试。同一合成脉冲在 1400 / 1200 MHz 参考频率下，与 v7.12 的原有 21 个检测字段逐项一致，仅新增 4 个 TOA 字段；同一份合成色散数据采用两种参考频率后，无穷大频率 TOA 在测试容差内一致。

本版的验证范围是单元测试与小规模合成数据，不以此声称已完成真实观测全量重跑。服务器测试在独立临时目录执行并已清理，没有覆盖正式流水线。
