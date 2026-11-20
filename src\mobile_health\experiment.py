import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dataset(n=2400,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,9));logit=.8*x[:,0]+.6*x[:,2]-.7*x[:,3]+.65*x[:,5]+.5*x[:,7]-1;y=r.binomial(1,1/(1+np.exp(-logit)));return x,y
def evaluate(seed=27):
 x,y=dataset(seed=seed);cut=1600;r=np.random.default_rng(seed+3);aug=x[:cut].copy();groups=[slice(0,3),slice(3,6),slice(6,9)];drop=r.integers(0,4,cut)
 for i,g in enumerate(groups,1):aug[drop==i,g]=0
 models={"standard":make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(x[:cut],y[:cut]),"modality_dropout":make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(np.vstack([x[:cut],aug]),np.r_[y[:cut],y[:cut]])};out={}
 for name,m in models.items():
  out[name]={}
  for condition,g in [("complete",None),("motion_missing",groups[0]),("audio_missing",groups[1]),("environment_missing",groups[2])]:
   xt=x[cut:].copy()
   if g is not None:xt[:,g]=0
   out[name][condition+"_auroc"]=float(roc_auc_score(y[cut:],m.predict_proba(xt)[:,1]))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
