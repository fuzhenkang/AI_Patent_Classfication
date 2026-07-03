# AI Patent Classification

本仓库用于根据专利标题和摘要识别 AI 专利，包含文本预处理、10 折交叉验证训练、模型评估和 Optuna 自动化参数寻优。

## 目录结构

```text
Preprocessing/preprocess_patents.py      # 标题和摘要预处理
DataSplit/create_cv_folds.py            # 生成分层 10 折 cv_fold 标记
Models/word2vec_cnn.py                  # Word2Vec + CNN
Models/word2vec_textcnn.py              # Word2Vec + TextCNN
Models/bert_cnn.py                      # BERT + CNN
Models/common.py                        # 公共数据、指标、标签编码和交叉验证工具
Evaluation/evaluate_model.py            # 对外部测试集或单个 fold 模型做补充评估
Optimization/optuna_search.py           # 基于 10 折交叉验证均值的 Optuna 参数寻优
data/stopwords/                         # 哈工大停用词表和用户补充词典
```

## 环境安装

```powershell
pip install -r requirements.txt
```

如果使用 GPU，建议根据本机 CUDA 版本安装对应的 PyTorch。

## 数据格式

原始 CSV 至少包含：

| 列名 | 含义 |
| --- | --- |
| `PN` | 专利申请号 |
| `title` | 专利标题 |
| `abstract` | 专利摘要 |

模型训练数据需要包含清洗后的文本列和标签列，默认列名为：

```text
text
label
```

如果标签列不是 `label`，运行脚本时用 `--label-col` 指定。

## 1. 文本预处理

```powershell
python Preprocessing\preprocess_patents.py `
  --input data\raw\patents.csv `
  --output data\processed\patents_cleaned.csv `
  --stats-output outputs\text_length_stats.csv `
  --keep-original-columns
```

预处理会生成：

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

## 2. 生成 10 折交叉验证标记

不再划分训练集、验证集和测试集。训练阶段采用分层 10 折交叉验证：按照类别比例将数据划分为 10 个规模相近且互不重叠的子集，每轮使用 9 折训练、1 折验证，最后取 10 次验证结果的平均值作为该组参数的验证性能。

可以先生成固定的 `cv_fold` 列，便于复现实验：

```powershell
python DataSplit\create_cv_folds.py `
  --input data\processed\patents_cleaned.csv `
  --output data\processed\patents_cleaned_cv.csv `
  --label-col label `
  --n-splits 10 `
  --seed 42
```

如果训练数据中没有 `cv_fold` 列，训练脚本也会根据 `label` 自动生成分层 10 折。

## 3. 10 折交叉验证训练

### Word2Vec + CNN

```powershell
python Models\word2vec_cnn.py `
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\word2vec_cnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --max-len 256 `
  --embedding-dim 200 `
  --num-filters 128 `
  --kernel-size 3 `
  --batch-size 64 `
  --epochs 10 `
  --lr 0.001
```

Word2Vec 词向量会在每一折内部基于该折训练数据自动训练。

### Word2Vec + TextCNN

```powershell
python Models\word2vec_textcnn.py `
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
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
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\bert_cnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --bert-model hfl/chinese-roberta-wwm-ext `
  --max-len 256 `
  --num-filters 128 `
  --kernel-sizes 3,4,5 `
  --batch-size 16 `
  --epochs 3 `
  --lr 0.00002
```

每个训练脚本会输出：

```text
outputs/<model_name>/cv_metrics.json
outputs/<model_name>/config.json
outputs/<model_name>/fold_00/
outputs/<model_name>/fold_01/
...
outputs/<model_name>/fold_09/
```

`cv_metrics.json` 中的 `accuracy`、`precision_macro`、`recall_macro`、`f1_macro`、`f1_weighted` 是 10 折验证结果的平均值。

## 4. Optuna 自动参数寻优

Optuna 以 10 折交叉验证平均指标作为目标函数，默认最大化 `f1_macro`。

### Word2Vec + CNN

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_cnn `
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\optuna\word2vec_cnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --n-trials 20
```

### Word2Vec + TextCNN

```powershell
python Optimization\optuna_search.py `
  --model-type word2vec_textcnn `
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\optuna\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --cv-folds 10 `
  --n-trials 20
```

### BERT + CNN

```powershell
python Optimization\optuna_search.py `
  --model-type bert_cnn `
  --data-csv data\processed\patents_cleaned_cv.csv `
  --output-dir outputs\optuna\bert_cnn `
  --text-col text `
  --label-col label `
  --bert-model hfl/chinese-roberta-wwm-ext `
  --cv-folds 10 `
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

## 5. 补充评估

如果后续有外部独立测试集，可以使用 `Evaluation/evaluate_model.py` 对某一个 fold 模型目录进行评估：

```powershell
python Evaluation\evaluate_model.py `
  --model-dir outputs\word2vec_cnn\fold_00 `
  --test-csv data\external_test.csv `
  --output-dir outputs\evaluation\word2vec_cnn_fold_00
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
-> DataSplit/create_cv_folds.py
-> Models/* 进行 10 折交叉验证训练
-> Optimization/optuna_search.py 基于 10 折均值寻优
```
