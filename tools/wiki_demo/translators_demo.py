import translators as ts

# Bing翻译（无需API密钥）, 但是 次数有限
# result = ts.translate_text("biceps, hypernym is [skeletal_muscle]", translator="bing", from_language='en',to_language='zh')
result = ts.translate_text("biceps, hypernym is [skeletal_muscle]", translator="Yandex", from_language='en',to_language='zh')

print(result)  # 输出：你好世界

