from nltk.corpus import wordnet as wn

print("="*40)
# 1. 查询同义词集
synsets = wn.synsets('good')
for syn in synsets:
    print(f"{syn.name()}: {syn.definition()}")

print("="*40)
# 2. 获取上位词
dog = wn.synset('dog.n.01')
hypernyms = dog.hypernyms()
print(hypernyms)

print("="*40)
# 3. 计算相似度
cat = wn.synset('cat.n.01')
similarity = dog.path_similarity(cat)  # 0.2
print(similarity)

print("="*40)
# 4. 词形还原
base_form = wn.morphy('running')  # 'run'
print(base_form)