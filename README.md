# 中文规则判定中的提示协议比较

本项目比较三种提示协议在固定合成中文许可/禁止规则集上的判定表现：直接判定（`direct`）、先输出紧凑规则关系再判定（`decompose`）和附加等长页面背景文字的控制条件（`length_control`）。规则初稿由语言模型辅助起草，经独立语义复核，并由程序枚举事实真值表；`gold_*` 标签是显式规则表达式的计算真值，不是专家标注。数据、提示文本、有效请求体、逐次 API 响应和离线分析均保存在仓库中。`gold_*` 字段仅用于离线评分，runner 只把场景、规则、禁止条款和事实发送给模型。

## 环境

- Python 3.11 或更新版本。
- Windows、macOS 或 Linux；实验代码使用标准库、`requests`、`numpy` 和 `matplotlib`。
- `CASDAO_API_KEY` 环境变量中提供 MaaS API 密钥。密钥不会作为命令参数、请求日志或仓库文件保存；runner 只在内存中读取它。

在项目根目录安装锁定范围内的依赖：

```powershell
python -m pip install -r requirements.txt
```

PowerShell 中在当前终端设置密钥后运行；不要把真实密钥写入脚本或配置：

```powershell
$env:CASDAO_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host "CASDAO_API_KEY" -AsSecureString)).Password
```

## 运行顺序

数据 JSONL 由 `src/build_dataset.py` 确定性生成；通常仓库已包含 `data/dataset.jsonl`。若重建，运行：

```powershell
python -m src.build_dataset
```

先检查开发任务数量，不发出网络请求：

```powershell
python -m src.runner --split dev --dry-run
```

开发集有12例，2个模型×3个提示条件共72个请求任务。开发运行：

```powershell
python -m src.runner --split dev
```

进程中断后必须加 `--resume`；已有成功 task ID 会去重，失败 task 会在不超过配置重试上限内继续。不要在已有日志上省略 `--resume`，否则会产生新的请求并增加预算消耗：

```powershell
python -m src.runner --split dev --resume
```

开发集检查并冻结提示、数据与评分流程后，再运行测试集：

```powershell
python -m src.runner --split test
```

测试集有48例，每个模型、提示条件和样本运行两次（本地重复编号17、43；不作为服务端 seed），共576个任务。测试中断后使用：

```powershell
python -m src.runner --split test --resume
```

所有任务完成后汇总两个 split；仅分析开发或测试可传 `--split dev` 或 `--split test`：

```powershell
python -m src.analyze --split all
```

runner 最多4并发；配置限定整个项目（包含3次模型探测）最多700次 chat-completions POST、450,000个 API 返回 token。`config/recovery.json` 若存在，只用于记录在案的 transport 恢复，不替代正式测试配置。每次尝试均记录 UTC 起止时间、任务 ID、精确请求体（不含 headers）、响应 JSON、返回模型、usage、finish reason、耗时和状态。认证错误响应正文不保存；日志字段在落盘前会精确替换当前密钥。

## 评分与分析

主正确率只读取严格 JSON 对象中固定的 `decision` 字段，且仅接受精确枚举 `ALLOW`/`DENY`。格式解析失败或无合法 decision 时主指标计错并单列为格式/决策字段错误；decision 合法但与 gold 不同则单列语义错误。严格 JSON、字段结构、简要依据长度、relation 长度及截断分别报告。围栏或前后文本中的首个合法 JSON 对象仅用于宽松解析敏感性指标；不从自由文本理由推测标签。

复杂测试题的配对差异先按 family 聚合三个样本和重复，再以 family 为 cluster 做5,000次配对 bootstrap。模型分别判断；不把两个模型样本合并。预设有用差异门槛为5个百分点；CI跨零表示证据不足。数据是预先指定的留出评估，不能作为自然业务规则总体效果的证明。

输出包括：

- `results/raw/attempts.jsonl`：每个实际请求尝试，包含重试和失败。
- `results/analysis/summary.json`、`metrics.csv`、`pairwise.csv`、`family_differences.csv`、`cases.json`、`costs.json`：汇总、配对估计、家族差异、错例/转变和请求资源账本。
- `figures/Fig1_accuracy.png`、`Fig2_complex_effects.png`、`Fig3_resources.png`：准确率、复杂题配对效应以及 token/延迟图；绘图数据保存在 `results/analysis/figure*_plot_data.csv`。
- `results/protocol_checks/`：启动前模型探测原始响应；4次 `/models` inventory GET 请求与 chat-completions POST 分开记账。

账本按 API 返回的 usage 统计输入、输出、缓存、推理 token（服务端提供时），并统计所有探测和重试的 POST。没有该兼容端点的可核验单价或账单时不推算人民币金额。若获得明确的 endpoint 单价，可提供 JSON 文件计算，单位均为人民币/百万 token：

```json
{
  "models": {
    "qwen3.8-flash": {
      "input_rmb_per_1m_tokens": 0.0,
      "cached_input_rmb_per_1m_tokens": 0.0,
      "output_rmb_per_1m_tokens": 0.0
    },
    "deepseek-v4.1-flash": {
      "input_rmb_per_1m_tokens": 0.0,
      "cached_input_rmb_per_1m_tokens": 0.0,
      "output_rmb_per_1m_tokens": 0.0
    }
  }
}
```

用真实核验的费率替换示例占位数后运行：

```powershell
python -m src.analyze --split all --prices config/prices.json
```

如果任一调用模型没有费率，或任一请求没有返回输入/输出 usage，分析会保留可计算的分项并省略整体金额。失败请求没有返回 usage 不代表其实际计费为零，返回 token 合计只是可观测部分。

## 离线检查

```powershell
python -m pytest -q
python -m unittest discover -s tests -v
```

测试只覆盖离线 prompt、评估、bootstrap、数据标签真值表、task 去重与预算逻辑，不发出 API 请求。
