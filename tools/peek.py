"""Print real tasks side by side: python tools/peek.py challenges.json cov.json N [seed] [maxdim]"""
import json, sys, random
import numpy as np
d = json.load(open(sys.argv[1])); cov = json.load(open(sys.argv[2]))
n = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
maxdim = int(sys.argv[5]) if len(sys.argv) > 5 else 14
unc = [k for k, v in cov.items() if not v]
random.Random(seed).shuffle(unc)
shown = 0
for k in unc:
    P = d[k]["train"]
    if max(max(np.array(p["input"]).shape + np.array(p["output"]).shape) for p in P) > maxdim:
        continue
    print("=" * 30, k)
    for p in P[:2]:
        a, b = np.array(p["input"]), np.array(p["output"])
        A = ["".join(map(str, r)) for r in a]; B = ["".join(map(str, r)) for r in b]
        for i in range(max(len(A), len(B))):
            print((A[i] if i < len(A) else " " * a.shape[1]) + "   " + (B[i] if i < len(B) else ""))
        print()
    shown += 1
    if shown >= n:
        break
