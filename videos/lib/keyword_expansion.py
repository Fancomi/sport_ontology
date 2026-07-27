"""关键词组合展开 —— 赛事型领域 (网球/羽毛球/其他隔网或对抗类比赛) 的通用召回放大器。

阶段一是多轮筛选的最上游: 候选池越大, 后面 (标题黑名单 → 缩略图 VLM → 真实帧 VLM)
才有东西可筛。而「完整比赛录像」最强的两类查询信号是:
  1. 选手对阵: `A vs B` —— 命中的几乎都是整场录像, 而非教学/集锦;
  2. 赛事 × 年份 × 轮次: `Wimbledon 2024 final` —— 同理。

这两类词都是名单的组合, 手写不现实 (12 人两两组合就 66 条, 20 赛事×4年×3轮 240 条)。
本模块把组合展开做成引擎能力, 领域只声明名单 (见 lib/domains.py 的 match_rosters /
event_names / event_years / event_rounds 等字段), 展开在采集时进行:

  - 声明式: 新增赛事型领域只填名单, 不需要写生成脚本、不需要把结果粘贴进 keywords.txt;
  - 无文件漂移: keywords.txt 只放手写词, 组合词是配置的纯函数, 不存在「文件与配置不一致」;
  - 确定性: 固定顺序 + 去重, 同一配置每次展开结果逐字节一致 —— 这是 search/diverse
    的 progress 文件能续跑的前提 (任务集必须可复现);
  - 向后兼容: 未声明名单的领域 (健身/羽毛球, 词表已在文件里且已跑过 progress)
    展开结果为空, 行为零变化。

若需人工抽查完整词表: `DOMAIN=<domain> python3 1_1_crawl.py dump-keywords`。
"""
from itertools import combinations


def _clean(values):
    """去空白 + 丢空串 + 保序去重 (展开的每一层都用它, 保证输出确定且无脏词)。"""
    out, seen = [], set()
    for v in values or ():
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def expand_matchups(rosters, templates):
    """选手对阵: 每个名单组内两两组合, 不跨组。

    分组的意义: 男单只和男单打, 女单只和女单打 —— 跨组配对会生成不存在的比赛,
    白耗请求配额。rosters 是「组」的序列, 每组是该项目的选手名单。
    templates 用 {a}/{b} 占位, 可给多个 (中英文不同写法各自成词)。
    """
    templates = _clean(templates)
    if not templates:
        return []
    pairs, seen_pairs = [], set()
    for group in rosters or ():
        members = _clean(group)
        for a, b in combinations(members, 2):
            key = frozenset((a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append((a, b))
    return _clean(t.format(a=a, b=b) for a, b in pairs for t in templates)


def expand_events(event_names, event_years, event_rounds, templates):
    """赛事 × 年份 × 轮次全叉乘。

    模板不含 {round} 时不按轮次重复展开 (例如「{event} {year} full match」这种
    整赛事查询), 由 _clean 的保序去重自然收敛。
    """
    events, years = _clean(event_names), _clean(event_years)
    rounds, templates = _clean(event_rounds), _clean(templates)
    if not (events and years and templates):
        return []
    # 模板不含 {round} 时给一个占位轮次, 展开后由去重收敛成单条
    rounds = rounds or [""]
    return _clean(
        t.format(event=ev, year=yr, round=rd)
        for ev in events for yr in years for rd in rounds for t in templates
    )


def expand_domain_keywords(domain):
    """按领域配置展开全部组合关键词 (对阵 + 赛事); 未声明名单的领域返回空。"""
    generated = expand_matchups(
        getattr(domain, "match_rosters", ()) or (),
        getattr(domain, "matchup_templates", ()) or (),
    )
    generated += expand_events(
        getattr(domain, "event_names", ()) or (),
        getattr(domain, "event_years", ()) or (),
        getattr(domain, "event_rounds", ()) or (),
        getattr(domain, "event_templates", ()) or (),
    )
    return _clean(generated)


def merge_keywords(file_keywords, generated_keywords):
    """手写词优先保序在前, 组合词追加在后, 全局去重。

    顺序有意义: search/diverse 的任务网格按关键词顺序生成, 手写词是人工挑过的
    高信号词, 放前面让它们在任务列表里靠前 (中断续跑时优先跑完)。
    """
    return _clean(list(file_keywords or ()) + list(generated_keywords or ()))
