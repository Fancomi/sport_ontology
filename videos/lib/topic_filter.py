"""话题门控 —— 阶段一的正向筛选 + 近邻运动排除 + 频道准入。

为什么需要它 (实测数据):
网球 clean 后 66.8 万条候选里, 标题含网球相关词的只有 34.3%; 时长分布也不像完整
比赛 (1-5 分钟占 41.4%, >60 分钟仅 6.6%)。抽样定位到两类系统性噪声:

  1. channel_crawl (25.6 万条): run_channels 从搜索结果里发现频道后会抓该频道
     **最近 200 个视频**, 不看内容。一个偶然出现在网球搜索结果里的综合频道
     (TMZ / GQ Taiwan / 新闻台) 会把它全部无关视频灌进候选池;
  2. 近邻运动: 乒乓球/匹克球/羽毛球/壁球/排球大量混入。且 "table tennis"、"匹克球"
     这些词本身就含 "tennis"/"球" —— title_blacklist 的子串黑名单挡不住 (会连
     真网球一起误杀), 必须「先排除近邻, 再判正向命中」。

设计取舍:
  - 门控是**规则层**, 不替代后面的缩略图 VLM / 真实帧 VLM 审核; 它只负责不让明显
    跑题的内容白占 GPU (阶段一是最上游, 但"大"不等于"什么都要");
  - 排除优先于包含: 近邻运动名里就含目标运动词, 顺序反了门控就失效;
  - 话题词从领域已声明的选手名单/赛事名自动派生, 不要求再手写一份平行名单 (避免漂移);
  - 种子频道无条件放行: 人工挑过的高质量来源 (Roland-Garros / Wimbledon 名字里没有
    "tennis") 不能被规则误杀;
  - 未配置话题词的领域 (健身/羽毛球) 门控整体关闭, 行为逐字节不变。
"""

# 派生词最短长度: 1-2 字符的拉丁碎片会命中任意标题, 让门控形同失效。
# CJK 单字信息密度远高于拉丁字母 ("网球"/"乒乓" 两字已足够特指), 故按字形分别设阈值:
# 含 CJK 的词按 _MIN_TERM_LEN_CJK, 纯拉丁/数字按 _MIN_TERM_LEN_LATIN。
_MIN_TERM_LEN_LATIN = 3
_MIN_TERM_LEN_CJK = 2


def _has_cjk(text):
    """是否含中日韩字符 (含中文、日文假名、韩文谚文)。"""
    for ch in text:
        code = ord(ch)
        if (0x3040 <= code <= 0x30FF          # 日文平假名/片假名
                or 0x3400 <= code <= 0x4DBF   # CJK 扩展 A
                or 0x4E00 <= code <= 0x9FFF   # CJK 基本区 (中文/日文汉字)
                or 0xAC00 <= code <= 0xD7AF   # 韩文谚文音节
                or 0x1100 <= code <= 0x11FF): # 韩文字母
            return True
    return False


def _min_len(term):
    return _MIN_TERM_LEN_CJK if _has_cjk(term) else _MIN_TERM_LEN_LATIN


# 分隔符归一化: 频道/标题里同一名字会写成 "Roland-Garros"/"Roland_Garros"/
# "Roland  Garros", 朴素子串匹配对不上空格写法的话题词。实测误伤过赛事官方频道
# (Roland-Garros), 也让 "Table-Tennis" 绕过了 "table tennis" 排除 —— 故匹配前把
# 连字符/下划线/连续空白一律折叠成单个空格 (词表与待匹配文本同样处理)。
_SEPARATORS = "-_—–/|.,:;!?()[]{}\"'`~+*@#$%^&=<>\\"


def _normalize_text(text):
    text = (text or "").lower()
    for sep in _SEPARATORS:
        text = text.replace(sep, " ")
    return " ".join(text.split())


def _norm_terms(values):
    """小写 + 分隔符归一 + 丢空串/过短 + 保序去重 (过短阈值按字形区分, 见 _min_len)。"""
    out, seen = [], set()
    for v in values or ():
        v = _normalize_text(v)
        if not v or len(v.replace(" ", "")) < _min_len(v) or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return tuple(out)


def _derive_person_terms(rosters):
    """从选手名单派生话题词: 全名 + 足够长的姓/名片段。

    只取全名会漏掉「Djokovic 2024 final」这种只写姓的标题; 只取片段会引入
    "open"/"cup" 之类通用词, 所以按最短长度阈值过滤后再收。
    """
    terms = []
    for group in rosters or ():
        for person in group or ():
            person = (person or "").strip()
            if not person:
                continue
            terms.append(person)
            # 英文名按空格切分取各段 (中文名不含空格, 整体已在上一行加入)
            for part in person.split():
                if len(part) >= _min_len(part):
                    terms.append(part)
    return terms


def build_topic_terms(domain):
    """领域的正向话题词全集 = 显式声明的 topic_include_terms + 从名单/赛事派生。

    未声明 topic_include_terms 的领域返回空元组 (门控关闭)。派生只是补充: 赛事型
    领域已经为关键词展开声明了选手名单和赛事名, 这里直接复用同一份配置。
    """
    explicit = getattr(domain, "topic_include_terms", ()) or ()
    if not explicit:
        return ()
    derived = _derive_person_terms(getattr(domain, "match_rosters", ()) or ())
    derived += list(getattr(domain, "event_names", ()) or ())
    return _norm_terms(list(explicit) + derived)


def compile_topic_gate(include_terms, exclude_terms):
    """把词表预先规范化成可复用的匹配器 (include, exclude) 元组。

    为什么必须预编译: run_clean 要对数十万条逐条判定, 若每次调用都重新
    _norm_terms(两份词表), 就是 O(行数 × 词数) 的重复规范化 —— 实测 66.8 万行 ×
    约 260 个词直接把 clean 拖到不可用。调用方应在循环外编译一次。
    """
    return _norm_terms(include_terms), _norm_terms(exclude_terms)


def topic_matches_compiled(title, channel, compiled):
    """已编译词表的判定 (循环内用这个; 语义与 topic_matches 完全一致)。"""
    include, exclude = compiled
    if not include:
        return True
    haystack = _normalize_text(f"{title or ''} {channel or ''}")
    for term in exclude:
        if term in haystack:
            return False
    return any(term in haystack for term in include)


def topic_matches(title, channel, include_terms, exclude_terms):
    """标题/频道是否属于目标话题 (便捷入口, 每次调用都会重新编译词表)。

    include_terms 为空 -> 门控关闭, 一律通过 (连 exclude 也不生效: 避免"只排除
    不包含"的半开状态悄悄改变旧领域口径)。
    否则: 先看 exclude (近邻运动命中即否), 再看 include (标题或频道命中即是)。

    批量场景请改用 compile_topic_gate + topic_matches_compiled, 见前者的说明。
    """
    return topic_matches_compiled(
        title, channel, compile_topic_gate(include_terms, exclude_terms))


def channel_allowed(channel, seed_channels, include_terms, exclude_terms, required):
    """频道是否准入 channel_crawl。

    required=False 或没有话题词 -> 全部放行 (旧行为)。
    种子频道无条件放行 (人工挑过, 名字未必含话题词)。
    其余频道: 频道名必须自身通过话题门控 —— 综合频道的最近 200 个视频基本跑题,
    与其抓回来再靠 VLM 逐个否掉, 不如不抓。
    """
    include = _norm_terms(include_terms)
    if not required or not include:
        return True
    if channel in (seed_channels or set()):
        return True
    return topic_matches("", channel, include, exclude_terms)
