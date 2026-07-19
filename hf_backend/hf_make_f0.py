import sys, csv, numpy as np
rows=[]
with open(sys.argv[1], newline="") as f:
    r=csv.reader(f); next(r)
    for row in r:
        if len(row)>=2: rows.append(float(row[1]))
np.save(sys.argv[2], np.asarray(rows, dtype=np.float32))
print(f"f0.npy: {len(rows)} frames")
