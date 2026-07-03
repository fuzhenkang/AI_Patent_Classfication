# AI Patent Classification

本仓库用于构建 AI 专利文本分类流程，当前包含专利标题和摘要的预处理脚本。

## 输入数据格式

CSV 文件至少需要包含以下三列：

| 列名 | 含义 |
| --- | --- |
| `PN` | 专利申请号 |
| `title` | 专利标题 |
| `abstract` | 专利摘要 |

如果原始文件没有表头，可以通过 `--no-header` 指定前三列为 `PN,title,abstract`。

## 文本预处理

脚本会完成以下操作：

- 去除 HTML/XML 标记、URL、邮箱、无法识别公式、异常控制字符。
- 统一中英文空格和标点，去除多余空格。
- 默认使用哈工大中文停用词表 `data/stopwords/hit_stopwords.txt`。
- 使用用户补充停用词表 `data/stopwords/user_stopwords.txt` 清理语料中未登录或未覆盖的停用词。
- 清理专利文本固定表述 `data/stopwords/patent_phrases.txt`，例如“本发明”“本实施例”“一种”等。
- 拼接清洗后的标题和摘要，生成 `text` 字段。
- 统计拼接文本长度，输出描述性统计表。

## 词典文件

| 文件 | 作用 |
| --- | --- |
| `data/stopwords/hit_stopwords.txt` | 哈工大中文停用词表，默认加载 |
| `data/stopwords/user_stopwords.txt` | 用户补充停用词表，每行一个词或短语 |
| `data/stopwords/patent_phrases.txt` | 专利固定表达词典，每行一个词或短语 |

`user_stopwords.txt` 会同时参与停用词过滤和短语级删除，适合补充语料中反复出现但哈工大表未覆盖的词，例如专利固定描述、领域无关泛化词等。

## 使用方式

```powershell
pip install -r requirements.txt

python src/preprocess_patents.py `
  --input data/raw/patents.csv `
  --output data/processed/patents_cleaned.csv `
  --stats-output outputs/text_length_stats.csv
```

如果 CSV 没有表头：

```powershell
python src/preprocess_patents.py `
  --input data/raw/patents.csv `
  --output data/processed/patents_cleaned.csv `
  --stats-output outputs/text_length_stats.csv `
  --no-header
```

如需指定其他哈工大停用词表或用户补充词典：

```powershell
python src/preprocess_patents.py `
  --input data/raw/patents.csv `
  --output data/processed/patents_cleaned.csv `
  --stats-output outputs/text_length_stats.csv `
  --hit-stopwords-file data/stopwords/hit_stopwords.txt `
  --user-stopwords-file data/stopwords/user_stopwords.txt
```

如需追加更多停用词文件，可以多次传入 `--stopwords-file`：

```powershell
python src/preprocess_patents.py `
  --input data/raw/patents.csv `
  --output data/processed/patents_cleaned.csv `
  --stats-output outputs/text_length_stats.csv `
  --stopwords-file data/stopwords/custom_stopwords.txt
```

## 输出字段

| 字段 | 含义 |
| --- | --- |
| `PN` | 专利申请号 |
| `title` | 原始标题 |
| `abstract` | 原始摘要 |
| `clean_title` | 清洗后的标题 |
| `clean_abstract` | 清洗后的摘要 |
| `text` | 清洗后标题和摘要的拼接文本 |
| `text_len_chars` | 拼接文本字符数，含空格 |
| `text_len_no_space` | 拼接文本字符数，不含空格 |
