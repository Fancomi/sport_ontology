"""Schema Registry: 版本化的 Key 集合。Key 只增不物理删(合并=软删)。

落盘: <dir>/schema/vN.json (整份快照) + <dir>/schema/HEAD (当前版本号)。
主键是稳定 ID k_NNN, 不随 name 变。
"""
import json
from pathlib import Path


class SchemaRegistry:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "schema"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keys: dict[str, dict] = {}
        self.version = -1
        self._load_head()

    # ── 持久化 ────────────────────────────────────────────
    def _head_file(self) -> Path:
        return self.dir / "HEAD"

    def _load_head(self) -> None:
        head = self._head_file()
        if head.exists():
            self.version = int(head.read_text().strip())
            data = json.loads((self.dir / f"v{self.version}.json").read_text("utf-8"))
            self.keys = {k["id"]: k for k in data["keys"]}

    def snapshot(self) -> int:
        """写下一版快照, 更新 HEAD, 返回新版本号。"""
        self.version += 1
        data = {"version": self.version, "keys": list(self.keys.values())}
        (self.dir / f"v{self.version}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        self._head_file().write_text(str(self.version))
        return self.version

    # ── Key 操作 ──────────────────────────────────────────
    def _next_id(self) -> str:
        return f"k_{len(self.keys):03d}"

    def add_key(self, *, name: str, desc: str, value_type: str,
                introduced_round: int, introduced_by: str,
                allowed_values: list | None = None,
                parent: str | None = None) -> str:
        kid = self._next_id()
        self.keys[kid] = {
            "id": kid, "name": name, "desc": desc, "value_type": value_type,
            "allowed_values": allowed_values or [], "parent": parent,
            "synonyms_of": None,
            "introduced_round": introduced_round, "introduced_by": introduced_by,
        }
        return kid

    def merge_key(self, kid: str, *, into: str) -> None:
        """软删 kid, 指向 into。历史引用不受影响。"""
        self.keys[kid]["synonyms_of"] = into

    def active_keys(self) -> list[dict]:
        return [k for k in self.keys.values() if k["synonyms_of"] is None]

    def n_active(self) -> int:
        return len(self.active_keys())

    def get(self, kid: str) -> dict | None:
        return self.keys.get(kid)
