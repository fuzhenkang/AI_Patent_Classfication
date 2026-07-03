# AI Patent Classification

本仓库用于根据专利标题和摘要识别 AI 专利，包含文本预处理、数据集划分、分类模型训练、独立评估和 Optuna 自动化参数寻优。

## 目录结构

```text
Preprocessing/preprocess_patents.py      # 标题和摘要预处理
DataSplit/split_dataset.py              # 8:1:1 划分训练集、验证集、测试集
Models/word2vec_cnn.py                  # Word2Vec + CNN
Models/word2vec_textcnn.py              # Word2Vec + TextCNN
Models/bert_cnn.py                      # BERT + CNN
Models/common.py                        # 公共数据集、指标、标签编码等工具
Evaluation/evaluate_model.py            # 独立测试集评估
Optimization/optuna_search.py           # Optuna 参数寻优
data/stopwords/                         # 哈工大停用词表和用户补充词典
```

## 环境安装

```powershell
pip install -r requirements.txt
```

如果使用 GPU，建议根据本机 CUDA 版本到 PyTorch 官网安装对应版本的 `torch`。

## 数据格式

原始 CSV 至少包含：

| 列名 | 含义 |
| --- | --- |
| `PN` | 专利申请号 |
| `title` | 专利标题 |
| `abstract` | 专利摘要 |

训练模型前，清洗后的数据还需要包含标签列，默认列名为 `label`，例如：

| PN | text | label |
| --- | --- | --- |
| CNxxxx | 人工智能 图像识别 模型 分类 | 1 |
| CNyyyy | 机械结构 控制 装置 | 0 |

如果你的标签列不是 `label`，运行脚本时用 `--label-col` 指定。

## 1. 文本预处理

```powershell
python Preprocessing\preprocess_patents.py `
  --input data\raw\patents.csv `
  --output data\processed\patents_cleaned.csv `
  --stats-output outputs\text_length_stats.csv `
  --keep-original-columns
```

预处理会清理特殊字符、多余空格、无法识别公式、停用词和专利固定表述，并生成：

```text
clean_title
clean_abstract
text
text_len_chars
text_len_no_space
```

默认使用：

```text
data/stopwords/hit_stopwords.txt
data/stopwords/user_stopwords.txt
data/stopwords/patent_phrases.txt
```

## 2. 数据集划分

将清洗后的数据划分为训练集、验证集和测试集，比例为 `8:1:1`：

```powershell
python DataSplit\split_dataset.py `
  --input data\processed\patents_cleaned.csv `
  --output-dir data\split `
  --text-col text `
  --label-col label `
  --seed 42
```

输出：

```text
data/split/train.csv
data/split/valid.csv
data/split/test.csv
data/split/split_summary.csv
```

## 3. 模型训练

### Word2Vec + CNN

```powershell
python Models\word2vec_cnn.py `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\word2vec_cnn `
  --text-col text `
  --label-col label `
  --max-len 256 `
  --embedding-dim 200 `
  --num-filters 128 `
  --kernel-size 3 `
  --batch-size 64 `
  --epochs 10 `
  --lr 0.001
```

Word2Vec 词向量会在训练脚本内部基于训练集自动训练，不需要单独训练。

### Word2Vec + TextCNN

```powershell
python Models\word2vec_textcnn.py `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --max-len 256 `
  --embedding-dim 200 `
  --num-filters 128 `
  --kernel-sizes 3,4,5 `
  --batch-size 64 `
  --epochs 10 `
  --lr 0.001
```

`--kernel-sizes 3,4,5` 表示同时使用 3、4、5 三种卷积窗口，每种窗口下有 `num_filters` 个卷积核。

### BERT + CNN

```powershell
python Models\bert_cnn.py `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\bert_cnn `
  --text-col text `
  --label-col label `
  --bert-model hfl/chinese-roberta-wwm-ext `
  --max-len 256 `
  --num-filters 128 `
  --kernel-sizes 3,4,5 `
  --batch-size 16 `
  --epochs 3 `
  --lr 0.00002
```

## 4. 独立测试集评估

训练完成后，用单独评估程序在测试集上评估：

```powershell
python Evaluation\evaluate_model.py `
  --model-dir outputs\word2vec_cnn `
  --test-csv data\split\test.csv `
  --output-dir outputs\evaluation\word2vec_cnn
```

输出：

```text
test_metrics.json
predictions.csv
```

将 `--model-dir` 换成 `outputs\word2vec_textcnn` 或 `outputs\bert_cnn` 即可评估其他模型。

## 5. Optuna 自动参数寻优

### Word2Vec + CNN 寻优

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_cnn `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\optuna\word2vec_cnn `
  --text-col text `
  --label-col label `
  --n-trials 20
```

### Word2Vec + TextCNN 寻优

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_textcnn `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\optuna\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --n-trials 20
```

### BERT + CNN 寻优

```powershell
python Optimization\optuna_search.py `
  --model-type bert_cnn `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\optuna\bert_cnn `
  --text-col text `
  --label-col label `
  --bert-model hfl/chinese-roberta-wwm-ext `
  --n-trials 10
```

寻优输出：

```text
optuna_trials.csv
best_params.json
trial_0000/
trial_0001/
...
```

默认最大化验证集 `f1_macro`，可通过 `--metric` 修改。

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
-> DataSplit/split_dataset.py
-> Models/*
-> Evaluation/evaluate_model.py
-> Optimization/optuna_search.py
```
