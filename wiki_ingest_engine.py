import os
import json
import subprocess
from tqdm import tqdm

# ==========================================
# CONFIGURATION: Ontology Slots
# ==========================================

SLOTS = ["exercise", "equipment", "body_part", "action", "phase", "laterality"]

# 提示词：用于从路径/文本中进行 5-Slot 抽取
# 注意：在实际生产中，这里会调用 Qwen3 并传入闭词表进行约束
EXTRACT_SLOT_PROMPT = """
你是一名动作识别领域的本体论专家。
请从给定的动作描述/路径中，根据【槽位定义】进行精确的属性抽取。

【槽位定义】:
1. exercise: 动作的大类 (例如: squat, lunge, press)
2. equipment: 使用的器械 (例如: dumbbell, barbell, none)
3. body_part: 涉及的身体部位 (例如: biceps, abdominals)
4. action: 具体的动作描述 (例如: extension, flexion, rotation)
5. phase: 动作的阶段 (例如: eccentric, concentric, isometric)

【输入数据】: {input_data}

【输出要求】:
请仅返回一个 JSON 对象，确保所有槽位都有值（若无法判定则填 null）。
严禁包含任何解释。

JSON 格式：
{{
  "exercise": "...",
  "equipment": "...",
  "body_part": "...",
  "action": "...",
  "phase": "...",
}}
"""

# ==========================================
# ENGINE
# ==========================================

class OntologyBuilder:
    def __init__(self, root_dir="ontology"):
        self.root_dir = root_dir
        self.slots_dir = os.path.join(root_dir, "schema")
        self.raw_dir = os.path.join(root_dir, "raw_extracted")
        self.stubs_dir = os.path.join(root_dir, "stubs")
        self.mapping_dir = os.path.join(root_dir, "mapping")
        
        self._setup_dirs()

    def _setup_dirs(self):
        """初始化目录结构"""
        for d in [self.slots_dir, self.raw_dir, self.stubs_dir, self.mapping_dir]:
            os.makedirs(d, exist_ok=True)

    def call_claude_extractor(self, input_data):
        """调用 LLM 进行槽位抽取"""
        prompt = EXTRACT_SLOT_PROMPT.format(input_data=input_data)
        try:
            # 这里模拟调用 Claude CLI
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True, text=True, check=True
            )
            raw_json = result.stdout.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw_json)
        except Exception as e:
            print(f"❌ Extraction Error: {e}")
            return None

    def update_slot_vocabulary(self, extracted_data):
        """
        将抽取的词汇更新到对应的槽位词表中 (Schema 层)
        这是构建本体树桩的第一步：收集所有标准词。
        """
        for slot in SLOTS:
            val = extracted_data.get(slot)
            if val and val != "null":
                slot_file = os.path.join(self.slots_dir, f"{slot}.json")
                
                # 读取现有词表
                vocab = []
                if os.path.exists(slot_file):
                    with open(slot_file, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)
                
                # 更新词表 (去重)
                if val not in vocab:
                    vocab.append(val)
                    with open(slot_file, 'w', encoding='utf-8') as f:
                        json.dump(vocab, f, indent=2, ensure_ascii=False)

    def process_all(self, exercise_list_file):
        """主循环：线性处理所有动作"""
        with open(exercise_list_file, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]

        print(f"🚀 开始构建 Ontology 树桩。总任务数: {len(paths)}")

        for path in tqdm(paths, desc="Ontology Building"):
            # 1. 抽取 (这里 input_data 可以是 path，也可以是解析后的文本)
            extracted_data = self.call_claude_extractor(path)
            
            if not extracted_data:
                continue

            # 2. 保存原始抽取结果 (Raw Extraction)
            # 使用路径的哈希或清洗后的名称作为文件名
            safe_name = path.split('/')[-1]
            raw_file = os.path.join(self.raw_dir, f"{safe_name}.json")
            with open(raw_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)

            # 3. 更新槽位词表 (Schema Layer)
            self.update_slot_vocabulary(extracted_data)

            # TODO: 后续步骤 - 关系发现 (Hierarchy/Confusion)

        print("✨ Ontology 树桩初步构建完成。")

if __name__ == "__main__":
    # 使用前请确保 all_exercises.txt 已存在
    builder = OntologyBuilder()
    builder.process_all("all_exercises.txt")
