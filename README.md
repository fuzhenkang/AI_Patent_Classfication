# AI Patent Classification

本仓库用于根据专利标题和摘要识别 AI 专利，实验流程为：先按 `8:2` 划分训练集和独立测试集，再在训练集内部进行 10 折交叉验证和 Optuna 参数寻优，随后使用最优参数在完整训练集上重训最终模型，最后在独立测试集上评估。

## 目录结构

```text
Preprocessing/preprocess_patents.py      # 标题和摘要预处理
DataSplit/split_train_test.py           # 按 8:2 分层划分训练集和测试集
DataSplit/create_cv_folds.py            # 对训练集生成分层 10 折 cv_fold 标记
Models/word2vec_cnn.py                  # Word2Vec + CNN
Models/word2vec_textcnn.py              # Word2Vec + TextCNN
Models/bert_cnn.py                      # BERT + CNN
Models/common.py                        # 公共数据、指标、标签编码和交叉验证工具
LLM/llm_lora_classifier.py              # AutoModelForSequenceClassification + LoRA
LLM/llm_registry.py                     # LLaMA/Qwen/GLM/Mistral/Baichuan 配置注册表
Optimization/optuna_search.py           # 基于训练集 10 折交叉验证均值的 Optuna 参数寻优
FinalTrain/train_best_model.py          # 使用最优参数在完整训练集上重训最终模型
Evaluation/evaluate_model.py            # 在独立测试集上评估最终模型
data/stopwords/                         # 哈工大停用词表和用户补充词典
```

## 环境安装

```powershell
pip install -r requirements.txt
```

如果使用 GPU，建议根据本机 CUDA 版本安装对应的 PyTorch。

## 数据格式

原始 CSV 至少包含 `PN`、`title`、`abstract` 三列。模型训练前，清洗后的数据还需要包含标签列，默认列名为 `label`。

## 1. 文本预处理

```powershell
python Preprocessing\preprocess_patents.py `
  --input data\raw\patents.csv `
  --output data\processed\patents_cleaned.csv `
  --stats-output outputs\text_length_stats.csv `
  --keep-original-columns
```

预处理会生成 `clean_title`、`clean_abstract`、`text`、`text_len_chars`、`text_len_no_space` 等字段。

## 2. 按 8:2 划分训练集和测试集

测试集只在最终模型确定后使用，不参与 10 折交叉验证和 Optuna 寻优。

```powershell
python DataSplit\split_train_test.py `
  --input data\processed\patents_cleaned.csv `
  --output-dir data\split `
  --label-col label `
  --train-ratio 0.8 `
  --seed 42
```

输出：

```text
data/split/train.csv
data/split/test.csv
data/split/train_test_summary.csv
```

## 3. 对训练集生成 10 折交叉验证标记

```powershell
python DataSplit\create_cv_folds.py `
  --input data\split\train.csv `
  --output data\split\train_cv.csv `
  --label-col label `
  --n-splits 10 `
  --seed 42
```

在训练集内部，脚本按照类别比例划分 10 个规模相近且互不重叠的子集。每轮选择 9 折训练、1 折验证，重复 10 次，并使用 10 次验证结果均值作为该组参数的验证性能。

## 4. Optuna 参数寻优

Optuna 默认最大化训练集 10 折交叉验证的 `f1_macro` 均值。

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_cnn `
  --data-csv data\split\train_cv.csv `
  --output-dir outputs\optuna\word2vec_cnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --n-trials 20
```

其他模型：

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_textcnn `
  --data-csv data\split\train_cv.csv `
  --output-dir outputs\optuna\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --n-trials 20
```

```powershell
python Optimization\optuna_search.py `
  --model-type bert_cnn `
  --data-csv data\split\train_cv.csv `
  --output-dir outputs\optuna\bert_cnn `
  --text-col text `
  --label-col label `
  --bert-model hfl/chinese-roberta-wwm-ext `
  --cv-folds 10 `
  --n-trials 10
```

输出：

```text
outputs/optuna/<model_name>/best_params.json
outputs/optuna/<model_name>/optuna_trials.csv
outputs/optuna/<model_name>/trial_0000/
...
```

## 5. 使用最优参数重训最终模型

以 `Word2Vec + CNN` 为例，读取 Optuna 的最佳参数，在完整 80% 训练集上重训最终模型：

```powershell
python FinalTrain\train_best_model.py `
  --model-type word2vec_cnn `
  --best-params outputs\optuna\word2vec_cnn\best_params.json `
  --train-csv data\split\train.csv `
  --output-dir outputs\final\word2vec_cnn `
  --text-col text `
  --label-col label
```

`Word2Vec + TextCNN`：

```powershell
python FinalTrain\train_best_model.py `
  --model-type word2vec_textcnn `
  --best-params outputs\optuna\word2vec_textcnn\best_params.json `
  --train-csv data\split\train.csv `
  --output-dir outputs\final\word2vec_textcnn `
  --text-col text `
  --label-col label
```

`BERT + CNN`：

```powershell
python FinalTrain\train_best_model.py `
  --model-type bert_cnn `
  --best-params outputs\optuna\bert_cnn\best_params.json `
  --train-csv data\split\train.csv `
  --output-dir outputs\final\bert_cnn `
  --text-col text `
  --label-col label `
  --bert-model hfl/chinese-roberta-wwm-ext
```

## 6. 基于大语言模型的 LoRA 微调分类

也可以直接使用 Hugging Face `AutoModelForSequenceClassification` 结合 LoRA 进行参数高效微调。当前支持通过注册表切换以下模型家族：

```text
llama
qwen
glm
mistral
baichuan
chinese_roberta
```

训练集 10 折交叉验证：

```powershell
python LLM\llm_lora_classifier.py `
  --model-key qwen `
  --data-csv data\split\train_cv.csv `
  --output-dir outputs\llm_lora\qwen `
  --text-col text `
  --label-col label `
  --max-len 256 `
  --batch-size 4 `
  --epochs 3 `
  --lr 0.00002 `
  --lora-r 8 `
  --lora-alpha 16 `
  --lora-dropout 0.1
```

如果默认模型不适合当前环境，可以用 `--base-model` 覆盖；如果 LoRA 目标模块不匹配，可以用 `--lora-target-modules` 覆盖。例如 Qwen/LLaMA/Mistral 常用 `q_proj,v_proj`，GLM 常用 `query_key_value`，Baichuan2 常用 `W_pack`。

基于 10 折交叉验证的 Optuna 寻优：

```powershell
python Optimization\optuna_search.py `
  --model-type qwen `
  --data-csv data\split\train_cv.csv `
  --output-dir outputs\optuna\qwen_lora `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --n-trials 10
```

在完整训练集上重训最终 LoRA 分类模型：

```powershell
python LLM\llm_lora_classifier.py `
  --model-key qwen `
  --train-csv data\split\train.csv `
  --output-dir outputs\final\qwen_lora `
  --text-col text `
  --label-col label `
  --max-len 256 `
  --batch-size 4 `
  --epochs 3 `
  --lr 0.00002 `
  --lora-r 8 `
  --lora-alpha 16 `
  --lora-dropout 0.1
```

## 7. 在测试集上评估最终模型

```powershell
python Evaluation\evaluate_model.py `
  --model-dir outputs\final\word2vec_cnn `
  --test-csv data\split\test.csv `
  --output-dir outputs\evaluation\word2vec_cnn `
  --text-col text `
  --label-col label
```

LoRA 分类模型测试：

```powershell
python Evaluation\evaluate_model.py `
  --model-dir outputs\final\qwen_lora `
  --test-csv data\split\test.csv `
  --output-dir outputs\evaluation\qwen_lora `
  --text-col text `
  --label-col label
```

输出：

```text
outputs/evaluation/<model_name>/test_metrics.json
outputs/evaluation/<model_name>/predictions.csv
```

## 常用默认参数

| 参数 | Word2Vec + CNN | Word2Vec + TextCNN | BERT + CNN |
| --- | --- | --- | --- |
| 最大文本长度 | 256 | 256 | 256 |
| 卷积核尺寸 | 3 | 3,4,5 | 3,4,5 |
| 每种尺寸卷积核个数 | 128 | 128 | 128 |
| batch size | 64 | 64 | 16 |
| epochs | 10 | 10 | 3 |
| learning rate | 0.001 | 0.001 | 0.00002 |
| dropout | 0.5 | 0.5 | 0.3 |
| 激活函数 | ReLU | ReLU | ReLU |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| 优化器 | AdamW | AdamW | AdamW |

## 推荐实验流程

```text
原始专利 CSV
-> Preprocessing/preprocess_patents.py
-> DataSplit/split_train_test.py
-> DataSplit/create_cv_folds.py
-> Optimization/optuna_search.py
-> FinalTrain/train_best_model.py
-> Evaluation/evaluate_model.py
```
