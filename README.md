# AI Patent Classification

本仓库用于构建 AI 专利文本分类流程。当前已包含专利标题和摘要的预处理脚本。

## 输入数据格式

CSV 文件至少需要包含以下三列：

| 列名 | 含义 |
| --- | --- |
| `PN` | 专利申请号 |
| `title` | 专利标题 |
| `abstract` | 专利摘要 |

如果原始文件没有表头，也可以通过命令行参数指定前三列为 `PN,title,abstract`。

## 文本预处理

脚本会完成以下操作：

- 去除 HTML/XML 标记、URL、邮箱、无法识别公式、异常控制字符。
- 统一中英文空格和标点，去除多余空格。
- 清理常见中文停用词。
- 清理专利文本中的固定表述，例如“本发明”“本实施例”“一种”等。
- 拼接清洗后的标题和摘要，生成 `text` 字段。
- 统计拼接文本长度，输出描述性统计表。

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

如需追加自定义停用词或专利固定表述，每行写一个词：

```powershell
python src/preprocess_patents.py `
  --input data/raw/patents.csv `
  --output data/processed/patents_cleaned.csv `
  --stats-output outputs/text_length_stats.csv `
  --stopwords-file data/stopwords.txt `
  --patent-phrases-file data/patent_phrases.txt
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

