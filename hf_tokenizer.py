from transformers import GPT2Tokenizer
import torch
from tokenizer import Tokenizer

# Load GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Use our own tokenizer
our_tokenizer = Tokenizer()

# English text
english_text = "Originated as the Imperial University of Peking in 1898, Peking University was China's first national comprehensive university and the supreme education authority at the time. Since the founding of the People's Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the '211 Project' and the '985 Project', the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."

# Chinese text
chinese_text = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"

# Train our tokenizer (using merged text as training data)
our_tokenizer.train(english_text + chinese_text, vocab_size=1024)

# Compare tokenization of English text
gpt2_tokens = gpt2_tokenizer.encode(english_text)
our_tokens = our_tokenizer.encode(english_text)

print("English text tokenization comparison:")
print(f"GPT-2 tokenizer length: {len(gpt2_tokens)}")
print(f"Our tokenizer length: {len(our_tokens)}")

# Compare tokenization of Chinese text
gpt2_tokens_zh = gpt2_tokenizer.encode(chinese_text)
our_tokens_zh = our_tokenizer.encode(chinese_text)

print("\nChinese text tokenization comparison:")
print(f"GPT-2 tokenizer length: {len(gpt2_tokens_zh)}")
print(f"Our tokenizer length: {len(our_tokens_zh)}") 