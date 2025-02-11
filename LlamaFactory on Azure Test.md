1. 安装conda（包含Python）
https://docs.anaconda.com/miniconda/
>添加环境变量
>C:\ProgramData\miniconda3
>C:\ProgramData\miniconda3\Scripts
***

2. ~~安装Python~~
[~~Python Release 3.13.0~~](https://www.python.org/downloads/release/python-3130/) 
***

3. 安装 Git（默认安装即可）
https://git-scm.com/downloads/win
***

4. 查看CUDA最高支持版本，并安装

	>参考：
	>[CUDA安装及环境配置](https://blog.csdn.net/chen565884393/article/details/127905428) 

	![剪贴板图片](/v2/file/notepad/downloadfile?file_id=26&location=2#size=800x414)

	>验证：nvcc -V
	>
	![剪贴板图片](/v2/file/notepad/downloadfile?file_id=27&location=2#size=800x128)
***

5. ~~安装PyTorch （先跳过，直接进行下一步）~~
版本确定 https://pytorch.org/get-started/locally/
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=29&location=2#size=400x228)
***

6. Clone LLaMa-Factory项目并安装依赖
先确定PyTorch版本 https://pytorch.org/get-started/locally/
版本不能高于CUDA安装版本！
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=30&location=2#size=800x306)

	```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git  #克隆项目
conda create -n llama_factory python=3.11 #创建conda项目环境
conda init #初始化conda 第一次使用需要 非第一使用不需要
conda activate llama_factory  #激活项目环境
cd LLaMA-Factory  #进入项目目录
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #安装pytorch 注意pytorch-cuda版本
pip install -e ".[torch,metrics]" #安装项目依赖
```

	**验证：**
```python
import torch  
torch.cuda.is_available()
torch.cuda.current_device()  
torch.cuda.get_device_name(0)  
torch.__version__
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=31&location=2#size=800x200)

	>**如果遇到以下错误：**
>*OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.*
>*OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMPDUPLICATELIBOK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.*
>
>**解决方法：**
>**查找重复libiomp5md.dll文件并删除其中一个，路径示例：C:\ProgramData\miniconda3\envs\llama_factory\Library\bin\libiomp5md.dll**

***
7. 下载模型
```bash
f:
cd AIModels
git clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct.git
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=32&location=2#size=800x344)

***
8. 原始模型直接推理
```bash
llamafactory-cli webchat ^
    --model_name_or_path F:\AIModels\Qwen2.5-1.5B-Instruct ^
    --template qwen
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=33&location=2#size=800x514)

***
9. 自定义数据集构建
>下载安装VS Code
>https://code.visualstudio.com/

	打开文件 F:\LLaMA-Factory\data\identity.json
	替换 {{name}} 和 {{author}} 后保存
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=34&location=2#size=800x512)
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=35&location=2#size=800x512)
	
	**自定义数据集** 则需要注册到 **dataset_info.json** 文件中.
	![剪贴板图片](/v2/file/notepad/downloadfile?file_id=36&location=2#size=800x286)
	
***
10. 基于LoRA的sft指令微调
>关于参数的完整列表和解释可以通过如下命令来获取
>llamafactory-cli train -h

	创建训练参数YAML
	F:\LLaMA-Factory\examples\train_lora\qwen_lora_sft.yaml
	
```python
### model
model_name_or_path: F:\AIModels\Qwen2.5-1.5B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: identity #identity,alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves\Qwen2.5-1.5B-Instruct\lora\train_identity_1
logging_steps: 10
save_steps: 50
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 10.0 #5.0 #3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
#ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

执行训练：
```bash
llamafactory-cli train F:\LLaMA-Factory\examples\train_lora\qwen_lora_sft.yaml
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=37&location=2#size=800x400)

Training Loss:
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=38&location=2#size=800x492)

***
11. 动态合并LoRA的推理
```bash
llamafactory-cli webchat 
    --model_name_or_path F:\AIModels\Qwen2.5-1.5B-Instruct 
    --adapter_name_or_path ./saves\Qwen2.5-1.5B-Instruct\lora\train_identity_1 
    --template qwen 
    --finetuning_type lora
```

![剪贴板图片](/v2/file/notepad/downloadfile?file_id=39&location=2#size=800x450)

***
12. 批量预测和训练效果评估
创建训练参数YAML
F:\LLaMA-Factory\examples\train_lora\qwen_lora_predict.yaml

```python
### model
model_name_or_path: F:\AIModels\Qwen2.5-1.5B-Instruct
adapter_name_or_path: saves\Qwen2.5-1.5B-Instruct\lora\train_identity_1

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: identity #,alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 20 #50
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/Qwen2.5-1.5B-Instruct/lora/predict_identity_1
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 4
predict_with_generate: true
#ddp_timeout: 180000000
```

执行评估：
```bash
llamafactory-cli train F:\LLaMA-Factory\examples\train_lora\qwen_lora_predict.yaml
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=40&location=2#size=800x508)


| 指标 | 含义 |
| -------- | -------- |
| BLEU-4 | BLEU（Bilingual Evaluation Understudy）是一种常用的用于评估机器翻译质量的指标。BLEU-4 表示四元语法 BLEU 分数，它衡量模型生成文本与参考文本之间的 n-gram 匹配程度，其中 n=4。值越高表示生成的文本与参考文本越相似，最大值为 100。  |
| predict_rouge-1 和 predict_rouge-2 | ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是一种用于评估自动摘要和文本生成模型性能的指标。ROUGE-1 表示一元 ROUGE 分数，ROUGE-2 表示二元 ROUGE 分数，分别衡量模型生成文本与参考文本之间的单个词和双词序列的匹配程度。值越高表示生成的文本与参考文本越相似，最大值为 100。|
| predict_rouge-l | ROUGE-L 衡量模型生成文本与参考文本之间最长公共子序列（Longest Common Subsequence）的匹配程度。值越高表示生成的文本与参考文本越相似，最大值为 100。|
| predict_runtime | 预测运行时间，表示模型生成一批样本所花费的总时间。单位通常为秒。 |
| predict_samples_per_second | 每秒生成的样本数量，表示模型每秒钟能够生成的样本数量。通常用于评估模型的推理速度。 |
| predict_steps_per_second	 | 每秒执行的步骤数量，表示模型每秒钟能够执行的步骤数量。对于生成模型，一般指的是每秒钟执行生成操作的次数。|

***
13. LoRA模型合并导出
创建导出参数YAML
```python
### model
model_name_or_path: F:\AIModels\Qwen2.5-1.5B-Instruct
adapter_name_or_path: saves\Qwen2.5-1.5B-Instruct\lora\train_identity_1
template: qwen
finetuning_type: lora
export_dir: F:\AIModels\Trained\Qwen2.5-1.5B-Instruct_lora_sft_identity_1
export_size: 2
export_device: cpu
export_legacy_format: False
```
执行导出：
```bash
llamafactory-cli export F:\LLaMA-Factory\examples\train_lora\qwen_lora_export.yaml
```
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=44&location=2#size=800x508)
![剪贴板图片](/v2/file/notepad/downloadfile?file_id=45&location=2#size=800x500)

***
13. 一站式 WebUI
>注意：目前webui版本只支持单机单卡，如果是多卡请使用命令行版本
>llamafactory-cli webui

![剪贴板图片](/v2/file/notepad/downloadfile?file_id=47&location=2#size=800x1330)

***
