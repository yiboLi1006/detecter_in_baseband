# Integrated Pipeline v3 —— VDIF/Mark5B 脉冲星检测流水线

## 概述

本流水线整合 **VDIF/Mark5B 数据读取 → 消色散（DM Correction）→ 脉冲检测 → CSV/PSRFITS/图像输出** 的全流程，适用于脉冲星单脉冲搜索。

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

```bash
python integrated_pipeline.py pipeline.ini
```

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
| `max_files` | int | `99999` | 最大输出文件数 |
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
| `csv_output_path` | str | `./pulse_csv_results` | CSV 输出路径（基础名，自动追加时间戳） |
| `output_raw_psrfits_dir` | str | `./pulse_psrfits_raw` | 原始 PSRFITS 文件输出目录 |
| `output_corrected_psrfits_dir` | str | `./pulse_psrfits_corrected` | DM 校正后 PSRFITS 文件输出目录 |
| `plot_output_dir` | str | 空 | 脉冲瀑布图输出目录；留空则禁用绘图 |

### [performance]
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cleanup_every_n_hdulists` | int | `50` | 每 N 个 hdulist 后回收 baseband 读取器（释放内部缓存）。设为 0 禁用 |

## 输出文件

每次检测到脉冲时，会生成以下文件：

```
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}.fits          # 原始 PSRFITS（仅含检测到脉冲的 hdulist）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_dm.fits       # DM 校正后 PSRFITS
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_segment.vdif   # 对应的基带数据段（如启用）
PSR_{源名}_{望远镜}_{文件编号:06d}_v{版本}_pulse{序号}.png # 脉冲瀑布图（如启用 plot_output_dir）
```

全部扫描结束后，在 `csv_output_path` 下生成带时间戳的 CSV 文件，包含所有检出脉冲的精确到达时间、幅度、FWHM、SNR 等信息。
