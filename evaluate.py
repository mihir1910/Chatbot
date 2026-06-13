import statistics

from rag import pipeline

GOLD = [
    ("What is the Betz limit for wind turbines?", "Renewable_Energy_Handbook.pdf"),
    ("How many kilocalories per gram do fats provide?", "Clinical_Nutrition_Guide.pdf"),
    ("What does TLS 1.3 use for forward secrecy?", "Network_Security_Manual.pdf"),
    ("What inflation rate do central banks often target?", "Macroeconomics_Primer.pdf"),
    ("How do mitochondria generate ATP?", "Cell_Biology_Notes.pdf"),
    ("What fraction of human carbon emissions do oceans absorb?", "Climate_Science_Report.pdf"),
    ("What problem is cache invalidation known to be?", "Software_Architecture_Text.pdf"),
    ("When did the Industrial Revolution begin and where?", "World_History_Survey.pdf"),
    ("What do beta-lactam antibiotics disrupt?", "Pharmacology_Compendium.pdf"),
    ("What escapes past the event horizon of a black hole?", "Astronomy_Lecture_Notes.pdf"),
]

K = 4


def main():
    lats, recs, rr, cites = [], [], [], []

    for q, gold in GOLD:
        res = pipeline.answer_query(q, final_k=K)
        lats.append(res["total_ms"])

        files = [r["filename"] for r in res.get("retrieved", [])]
        hit = gold in files
        recs.append(1.0 if hit else 0.0)

        rank = next((i + 1 for i, f in enumerate(files) if f == gold), None)
        rr.append(1.0 / rank if rank else 0.0)

        cites.append(1.0 if gold in res["answer"] else 0.0)

        mark = "✓" if hit else "✗"
        print(f"{mark} [{res['total_ms']}ms] {q[:55]:<55} -> {files[:2]}")

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Queries:            {len(GOLD)}")
    print(f"Latency p50:        {statistics.median(lats):.0f} ms")
    print(f"Latency p95:        {sorted(lats)[int(0.95*len(lats))-1]:.0f} ms")
    print(f"Latency max:        {max(lats):.0f} ms")
    print(f"Recall@{K}:           {statistics.mean(recs):.2f}")
    print(f"MRR:                {statistics.mean(rr):.2f}")
    print(f"Citation accuracy:  {statistics.mean(cites):.2f}")
    print("====================================================")
    n_ok = sum(1 for t in lats if t <= 5000)
    print(f"Within 5s budget:   {n_ok}/{len(lats)}")


if __name__ == "__main__":
    main()
