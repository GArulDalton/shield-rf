import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
os.chdir(os.path.join(ROOT, "runs"))
import json,glob,numpy as np,sys
part=sys.argv[1]
R=[json.load(open(f)) for f in sorted(glob.glob(f'res_{part}_*.json'))]
print(len(R),'seeds')
out={}
for m in R[0]:
    out[m]={}
    for tag in ['seen','unseen']:
        out[m][tag]={k:(float(np.mean([r[m][tag][k] for r in R])),float(np.std([r[m][tag][k] for r in R]))) for k in R[0][m][tag]}
    if 'train_s' in R[0][m]: out[m]['train_s']=float(np.mean([r[m]['train_s'] for r in R]))
json.dump(out,open(f'agg_{part}.json','w'),indent=1)
for m in out:
    for tag in ['seen','unseen']:
        d=out[m][tag]; print(f"{m:38s} {tag:6s} "+" ".join(f"{k}={100*v[0]:.1f}±{100*v[1]:.1f}" for k,v in d.items()))
