import opencc
converter = opencc.OpenCC('s2t.json')
print(converter.convert('汉字'))  # 漢字

converter = opencc.OpenCC('t2s.json')
print(converter.convert('漢字'))  # 汉字