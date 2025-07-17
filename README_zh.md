[**English Homepage**](https://github.com/modelscope/Trinity-RFT/blob/main/README.md) | [**Tutorial**](https://modelscope.github.io/Trinity-RFT/) | [**FAQ**](./docs/sphinx_doc/source/tutorial/faq.md)

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 120px;">
</div>



<h2 align="center">Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models</h2>


<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://modelscope.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>



## 🚀 最新动态

* [2025-07] Trinity-RFT v0.2.0 发布。
* [2025-07] 我们更新了[技术报告](https://arxiv.org/abs/2505.17826) (arXiv v2)，增加了新功能、示例和实验。
* [2025-06] Trinity-RFT v0.1.1 发布。
* [2025-05] 我们发布了 Trinity-RFT v0.1.0 和一份技术报告。
* [2025-04] Trinity-RFT 的初始代码库正式开源。


## 💡 Trinity-RFT 是什么？



Trinity-RFT是一个通用、灵活且易于使用的大语言模型强化学习微调（RFT）框架。
它旨在支持多样化的应用场景，并作为一个用于在[经验时代](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)探索先进的强化学习（RL）范式的平台。



## ✨ 核心特性

* **统一的 RFT 核心：**

  支持*同步/异步*（synchronous/asynchronous）、*在线策略/离线策略*（on-policy/off-policy）和*在线/离线*（online/offline）训练，数据产生（rollout）和训练（training）可以分别在不同设备上独立运行和扩展。

* **一流的智能体-环境交互：**

  优雅地处理滞后反馈、长尾延迟以及智能体/环境故障。支持智能体和环境之间的多轮交互。

* **优化的数据管道：**

  将数据收集任务和经验视为动态资产，支持在整个 RFT 生命周期中进行主动管理（如优先级排序、清洗、增强）。

* **用户友好的设计：**

  采用了模块化和解耦的架构，便于采纳和二次开发，并提供丰富的图形用户界面以支持低代码使用。


<p align="center">
  <img src="https://img.alicdn.com/imgextra/i2/O1CN01H3UbpF1yP7E1OCLbi_!!6000000006570-2-tps-1334-638.png" alt="Trinity-RFT">
  <em>图：Trinity-RFT 的设计</em>
</p>


<details>
<summary>图：RFT-core 的架构</summary>


<p align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01BFCZRV1zS9T1PoH49_!!6000000006712-2-tps-922-544.png" alt="Trinity-RFT-core-architecture">
</p>

</details>


<details>
<summary>图：Trinity-RFT 支持的部分 RFT 模式</summary>

<p align="center">
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="Trinity-RFT-modes">
</p>


</details>


<details>
<summary>图：数据处理器的架构</summary>

<p align="center">
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01hR1LCh25kpJMKmYR4_!!6000000007565-2-tps-1474-740.png" alt="Trinity-RFT-data-pipeline-buffer">
</p>

</details>


<details>
<summary>图：Trinity-RFT 中数据处理的设计</summary>

<p align="center">
  <img src="https://img.alicdn.com/imgextra/i4/O1CN01UvyfcZ1WoTv5t3pCp_!!6000000002835-2-tps-1166-274.png" alt="Trinity-RFT-data-pipelines">
</p>

</details>



## 🛠️ 我可以用 Trinity-RFT 做什么？


* **适应新场景：**

  只需在单个 `Workflow` 或 `MultiTurnWorkflow` 类中实现智能体-环境交互逻辑。([示例](./docs/sphinx_doc/source/tutorial/example_multi_turn.md))


* **强化学习算法开发：**

  在紧凑、即插即用的类中开发自定义的强化学习算法（包括损失函数设计、采样、数据处理等）。([示例](./docs/sphinx_doc/source/tutorial/example_mix_algo.md))


* **低代码使用：**

  使用图形化界面轻松监控和追踪学习过程。


---

## 目录


- [快速上手](#getting-started)
  - [第一步：安装](#step-1-installation)
  - [第二步：准备数据集和模型](#step-2-prepare-dataset-and-model)
  - [第三步：配置](#step-3-configurations)
  - [第四步：运行 RFT 流程](#step-4-run-the-rft-process)
- [更多教程](#further-tutorials)
- [未来功能](#upcoming-features)
- [贡献指南](#contribution-guide)
- [致谢](#acknowledgements)
- [引用](#citation)



## 快速上手


> [!NOTE]
> 本项目正处于活跃开发阶段。欢迎提出意见和建议！


### 第一步：安装


源码安装 **（推荐）**：

```shell
# 从 GitHub 拉取源码
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# 使用 Conda 或 venv 创建新环境
# 选项 1：Conda
conda create -n trinity python=3.10
conda activate trinity

# 选项 2：venv
python3.10 -m venv .venv
source .venv/bin/activate

# 以可编辑模式安装包
# 适用于 bash
pip install -e .[dev]
# 适用于 zsh
pip install -e .\[dev\]

# 安装完所有依赖后，再安装 flash-attn
# 注意：flash-attn 编译需要较长时间，请耐心等待。
# 适用于 bash
pip install -e .[flash_attn]
# 适用于 zsh
pip install -e .\[flash_attn\]
# 如果安装 flash-attn 时遇到错误，可以尝试以下命令
# pip install flash-attn -v --no-build-isolation
```

使用 pip 安装：

```shell
pip install trinity-rft==0.2.0
```

使用 Docker 安装：
我们为 Trinity-RFT (trinity) 提供了 Dockerfile

```shell
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# 构建 Docker 镜像
# 注意：您可以编辑 Dockerfile 来定制环境
# 例如，使用 pip 镜像或设置 API 密钥
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# 运行 Docker 镜像
docker run -it --gpus all --shm-size="64g" --rm -v $PWD:/workspace -v <root_path_of_data_and_checkpoints>:/data trinity-rft:latest
```


**环境要求：**
Python 版本 >= 3.10，
CUDA 版本 >= 12.4，
以及至少 2 块 GPU。


### 第二步：准备数据集和模型


Trinity-RFT 支持来自 Huggingface 和 ModelScope 的大多数数据集和模型。


**准备模型**，保存到本地目录 `$MODEL_PATH/{model_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}

# 使用 ModelScope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

更多关于模型下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 或  [ModelScope](https://modelscope.cn/docs/models/download)。



**准备数据集**，保存到本地目录 `$DATASET_PATH/{dataset_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}

# 使用 ModelScope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```

更多关于数据集下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) 或 [ModelScope](https://modelscope.cn/docs/datasets/download)。



### 第三步：配置


Trinity-RFT 提供了一个 Web 界面来配置您的 RFT 流程。

> [!NOTE]
> 这是一个实验性功能，我们将持续改进。


要启用最小功能（主要用于训练器），您可以运行

```bash
trinity studio --port 8080
```

然后您可以在网页上配置您的 RFT 流程并生成一个配置文件。您可以保存该配置以备后用，或按照下一节的描述直接运行。

高级用户也可以直接编辑配置文件。
我们在 [`examples`](examples/) 目录中提供了一些示例配置文件。

若需完整的 GUI 功能，请参考 [Trinity-Studio](https://github.com/modelscope/Trinity-Studio) 仓库。


<details>

<summary> 示例：配置管理器 GUI </summary>

![config-manager](https://img.alicdn.com/imgextra/i1/O1CN01yhYrV01lGKchtywSH_!!6000000004791-2-tps-1480-844.png)


</details>




### 第四步：运行 RFT 流程


启动一个 Ray 集群：

```shell
# 在主节点上
ray start --head

# 在工作节点上
ray start --address=<master_address>
```

（可选）登录 [wandb](https://docs.wandb.ai/quickstart/) 以便更好地监控：

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

对于命令行用户，运行 RFT 流程：

```shell
trinity run --config <config_path>
```

例如，以下是在 GSM8k 数据集上使用 GRPO 微调 Qwen2.5-1.5B-Instruct 的命令：

```shell
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

对于 Studio 用户，在 Web 界面中点击“运行”。


## 更多教程


运行不同 RFT 模式的教程：

+ [快速示例：在 GSM8k 上运行 GRPO](./docs/sphinx_doc/source/tutorial/example_reasoning_basic.md)
+ [离线策略 RFT](./docs/sphinx_doc/source/tutorial/example_reasoning_advanced.md)
+ [完全异步 RFT](./docs/sphinx_doc/source/tutorial/example_async_mode.md)
+ [通过 DPO 或 SFT 进行离线学习](./docs/sphinx_doc/source/tutorial/example_dpo.md)


将 Trinity-RFT 适配到新的多轮智能体场景的教程：

+ [多轮任务](./docs/sphinx_doc/source/tutorial/example_multi_turn.md)


数据相关功能的教程：

+ [高级数据处理与人机协同](./docs/sphinx_doc/source/tutorial/example_data_functionalities.md)


使用 Trinity-RFT 进行 RL 算法开发/研究的教程：

+ [使用 Trinity-RFT 进行 RL 算法开发](./docs/sphinx_doc/source/tutorial/example_mix_algo.md)


完整配置指南：请参阅[此文档](./docs/sphinx_doc/source/tutorial/trinity_configs.md)


给开发者和研究人员的指南：

+ [构建新的 RL 场景](./docs/sphinx_doc/source/tutorial/trinity_programming_guide.md#workflows-for-rl-environment-developers)
+ [实现新的 RL 算法](./docs/sphinx_doc/source/tutorial/trinity_programming_guide.md#algorithms-for-rl-algorithm-developers)




## 未来功能

路线图：[#51](https://github.com/modelscope/Trinity-RFT/issues/51)



## 贡献指南


本项目正处于活跃开发阶段，我们欢迎来自社区的贡献！


代码风格检查：

```shell
pre-commit run --all-files
```



单元测试：

```shell
python -m pytest tests
```



## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl) 和 [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) 用于数据处理管道；
+ [AgentScope](https://github.com/modelscope/agentscope) 用于智能体工作流；
+ [Ray](https://github.com/ray-project/ray) 用于分布式系统；
+ 我们也从 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL](https://github.com/huggingface/trl) 和 [ChatLearn](https://github.com/alibaba/ChatLearn) 等框架中汲取了灵感；
+ ......

## 引用


```plain
@misc{trinity-rft,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}
```
