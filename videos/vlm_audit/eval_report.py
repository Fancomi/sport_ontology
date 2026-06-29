#!/usr/bin/env python3
"""读 _experiments/result_*.json, 出 4 变体对比: 9负例召回 / 正例reject列表 / 耗时。

用法: python3 eval_report.py   # 自动读所有 result_<V>.json
"""
import os, sys, json, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(_HERE, "_experiments")
NEGATIVES = {"9uGbomnOApI_2", "Ffqz_nbe0mo_1", "mrTpGLyMboc_8", "0zg0MmFl2R8_19",
             "RrpZS_oX9QM_3", "Oxa8-kW8yyQ_17", "5a7fOvGOuAM_1", "rPJ88Oy4H8I_11", "4MdP56Mryrw_5"}


def analyze(result_path):
    data = json.load(open(result_path, encoding="utf-8"))
    neg_caught = [c for c in data if c in NEGATIVES and data[c]["verdict"] == "reject"]
    neg_missed = [c for c in NEGATIVES if c in data and data[c]["verdict"] != "reject"]
    pos_rejected = [c for c in data if c not in NEGATIVES and data[c]["verdict"] == "reject"]
    errs = [c for c in data if data[c]["verdict"] == "error"]
    avg_ms = round(sum(data[c].get("elapsed_ms", 0) for c in data) / max(1, len(data)))
    return {"n": len(data), "neg_caught": neg_caught, "neg_missed": neg_missed,
            "pos_rejected": pos_rejected, "errors": errs, "avg_ms": avg_ms}


def main():
    results = sorted(glob.glob(os.path.join(EXP, "result_*.json")))
    if not results:
        sys.exit("no result_*.json (run run_experiment.py first)")
    lines = ["# VLM 审核变体对比\n"]
    lines.append("| 变体 | 切片数 | 9负例召回 | 漏掉的负例 | 正例被reject数 | error | 均耗时ms | 全量196万预估h |")
    lines.append("|---|---|---|---|---|---|---|---|")
    detail = {}
    for rp in results:
        v = os.path.basename(rp)[len("result_"):-len(".json")]
        a = analyze(rp); detail[v] = a
        full_h = round(a["avg_ms"] * 1961084 / 1000 / 3600 / 4, 1)  # 4 端点并行粗估
        lines.append(f"| {v} | {a['n']} | {len(a['neg_caught'])}/9 | "
                     f"{','.join(a['neg_missed']) or '-'} | {len(a['pos_rejected'])} | "
                     f"{len(a['errors'])} | {a['avg_ms']} | ~{full_h} |")
    lines.append("\n## 各变体「正例被 reject」明细 (交人工核查: 是误杀还是真负例)\n")
    for v, a in detail.items():
        lines.append(f"### {v}  (reject {len(a['pos_rejected'])} 个默认正例)")
        lines.append("```\n" + "\n".join(a["pos_rejected"]) + "\n```" if a["pos_rejected"] else "(无)")
    md = "\n".join(lines)
    open(os.path.join(EXP, "compare.md"), "w", encoding="utf-8").write(md)
    json.dump(detail, open(os.path.join(EXP, "compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(md)
    print(f"\n-> {os.path.join(EXP, 'compare.md')}")


if __name__ == "__main__":
    main()
