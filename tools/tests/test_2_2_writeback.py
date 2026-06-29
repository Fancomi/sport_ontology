# tools/tests/test_2_2_writeback.py
import sys, os, importlib, json, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_2_translate_augment')

CN_FILE = 'augment_front_cn.json'
EN_FILE = 'augment_front_en.json'
F = 'category_3_slotted_description'


def _setup(tmp, cn_cat3, en_seed):
    """建临时动作目录：CN(含新键) + EN定版(失效态:无_validated/无cat3 让其走translate)。"""
    (tmp / 'metadata_cn.json').write_text(json.dumps({'exercise': '测试'}, ensure_ascii=False), 'utf-8')
    (tmp / CN_FILE).write_text(json.dumps({F: cn_cat3,
        'category_1_visual_description': '', 'category_2_sports_guidance': {}}, ensure_ascii=False), 'utf-8')
    (tmp / EN_FILE).write_text(json.dumps(en_seed, ensure_ascii=False), 'utf-8')


class _MisalignClient:
    """翻译返回键集与 CN 不一致的译文；QC 一律 pass（模拟 LLM 软判失灵）。"""
    def chat(self, messages, **kw):
        sys_txt = messages[0]['content']
        if 'quality checker' in sys_txt or '质检' in sys_txt or 'checker' in sys_txt.lower():
            return '{"pass": true}'                       # QC 谎报 pass
        # 翻译：丢掉 force_type（键集比 CN 少一个）
        return json.dumps({F: '[gender:male] pulls',
            'category_1_visual_description': 'x', 'category_2_sports_guidance': {}}, ensure_ascii=False)


def test_process_one_does_not_write_misaligned_translation():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # CN 有 gender+force_type；EN 定版态(失效)：无 cat3、无 _validated → 触发 translate
        _setup(tmp, '[gender:男]通过[force_type:拉]发力',
               {F: '', '_translated': True})   # 失效态(无 _validated 无 cat3)
        before = (tmp / EN_FILE).read_text('utf-8')
        mod.process_one(tmp / 'metadata_cn.json', _MisalignClient(), do_qc=True)
        after_d = json.loads((tmp / EN_FILE).read_text('utf-8'))
        # 键集不齐 → QC 必判 False(确定性gate) → 不得写入未对齐译文、不得打 _validated
        assert not after_d.get('_validated'), 'QC未通过却打了_validated'
        assert not after_d.get(F), '未对齐译文不应被写盘'
