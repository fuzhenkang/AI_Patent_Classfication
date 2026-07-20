# AI Patent Classification

本仓库用于根据专利标题、摘要等文本信息识别 AI 专利。

## 项目结构

```text
preprocess_patents.py     文本预处理：清洗标题和摘要、去停用词、拼接文本、统计文本长度
split_train_test.py       按类别比例划分 train.csv、valid.csv、test.csv
optuna_search.py          基于训练集和验证集进行 Optuna 参数寻优
train_best_model.py       使用 Optuna 最优参数重新训练最终模型
evaluate_model.py         在测试集上评估最终模型
requirements.txt          Python 依赖
models/                   Word2Vec+CNN、Word2Vec+TextCNN、BERT+CNN、BERT线性分类头及公共工具
stopwords/                哈工大停用词、专利固定表述和用户补充停用词
data/                     预留数据目录，用于后续存放原始数据、清洗数据和划分数据
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

如果使用 GPU，建议根据本机 CUDA 版本安装对应的 PyTorch。

## 数据格式

原始 CSV 至少包含以下列：

```csv
PN,title,abstract,label
CNxxxx,一种图像识别方法,本发明公开了一种基于神经网络的图像识别方法,1
CNyyyy,一种机械连接装置,本发明涉及机械零件连接结构,0
```

其中：

```text
PN        专利申请号
title     专利标题
abstract  专利摘要
label     分类标签，默认 0/1，也可以是文字标签
```

## 1. 文本预处理

```powershell
python preprocess_patents.py \
  --input data\raw\patents.csv \
  --output data\processed\patents_cleaned.csv \
  --stats-output data\processed\text_length_stats.csv v
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

默认停用词文件位于：

```text
stopwords/hit_stopwords.txt
stopwords/patent_phrases.txt
stopwords/user_stopwords.txt
```

## 2. 划分训练集、验证集和测试集

```powershell
python split_train_test.py \
  --input data\processed\patents_cleaned.csv v
  --output-dir data\split v
  --label-col label \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

输出：

```text
data/split/train.csv
data/split/valid.csv
data/split/test.csv
data/split/split_summary.csv
```

当前项目采用固定训练集、验证集和测试集流程，不再使用交叉验证训练。

## 3. Optuna 参数寻优

以 Word2Vec + CNN 为例：

```powershell
python optuna_search.py \
  --model-type word2vec_cnn \
  --train-csv data\split\train.csv \
  --valid-csv data\split\valid.csv \
  --output-dir outputs\optuna\word2vec_cnn \
  --text-col text \
  --label-col label \
  --n-trials 20
```

Word2Vec + TextCNN：

```powershell
python optuna_search.py `
  --model-type word2vec_textcnn `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\optuna\word2vec_textcnn `
  --text-col text `
  --label-col label `
  --n-trials 20
```

BERT + CNN：

```powershell
python optuna_search.py \
  --model-type bert_cnn \
  --train-csv data\split\train.csv \
  --valid-csv data\split\valid.csv \
  --output-dir outputs\optuna\bert_cnn \
  --text-col text \
  --label-col label \
  --bert-model hfl/chinese-roberta-wwm-ext \
  --n-trials 10
```

BERT 线性分类头：

```powershell
python optuna_search.py \
  --model-type bert_linear \
  --train-csv data\split\train.csv \
  --valid-csv data\split\valid.csv \
  --output-dir outputs\optuna\bert_linear \
  --text-col text \
  --label-col label \
  --bert-model hfl/chinese-roberta-wwm-ext \
  --n-trials 10
```

输出：

```text
outputs/optuna/<model_name>/best_params.json
outputs/optuna/<model_name>/optuna_trials.csv
outputs/optuna/<model_name>/trial_0000/
```

## 4. 使用最优参数训练最终模型

Word2Vec + CNN：

```powershell
python train_best_model.py \
  --model-type word2vec_cnn \
  --best-params outputs\optuna\word2vec_cnn\best_params.json \
  --train-csv data\split\train.csv \
  --output-dir outputs\final\word2vec_cnn \
  --text-col text \
  --label-col label
```

Word2Vec + TextCNN：

```powershell
python train_best_model.py \
  --model-type word2vec_textcnn \
  --best-params outputs\optuna\word2vec_textcnn\best_params.json \
  --train-csv data\split\train.csv \
  --output-dir outputs\final\word2vec_textcnn \
  --text-col text \
  --label-col label
```

BERT + CNN：

```powershell
python train_best_model.py \
  --model-type bert_cnn \
  --best-params outputs\optuna\bert_cnn\best_params.json \
  --train-csv data\split\train.csv \
  --output-dir outputs\final\bert_cnn \
  --text-col text \
  --label-col label \
  --bert-model hfl/chinese-roberta-wwm-ext
```

BERT 线性分类头：

```powershell
python train_best_model.py \
  --model-type bert_linear `
  --best-params outputs\optuna\bert_linear\best_params.json \
  --train-csv data\split\train.csv \
  --output-dir outputs\final\bert_linear \
  --text-col text \
  --label-col label \
  --bert-model hfl/chinese-roberta-wwm-ext
```

## 5. 测试集评估

```powershell
python evaluate_model.py \
  --model-dir outputs\final\word2vec_cnn \
  --test-csv data\split\test.csv \
  --output-dir outputs\evaluation\word2vec_cnn \
  --text-col text \
  --label-col label
```

输出：

```text
outputs/evaluation/<model_name>/test_metrics.json
outputs/evaluation/<model_name>/predictions.csv
```

## 支持的模型

```text
models/word2vec_cnn.py      Word2Vec + CNN
models/word2vec_textcnn.py  Word2Vec + TextCNN
models/bert_cnn.py          BERT + CNN
models/bert_linear.py       BERT + 线性分类头，基于 AutoModelForSequenceClassification
```
