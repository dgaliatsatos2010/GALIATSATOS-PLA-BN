from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from plabn import GaliatsatosMethod, coarsening_rank
from publication_experiments import OperatorAwareLogisticEM
from nhanes_diabetes_validation import metrics, random_environment_split
K=3
METHODS=["GALIATSATOS/PLA-BN TAN","Operator-aware logistic EM","Oracle-label logistic"]
METRICS=["accuracy","balanced_accuracy","macro_f1","log_loss","brier"]
OPERATORS=[np.array([[1.,0.,0.],[0.,1.,1.]]),np.array([[1.,1.,0.],[0.,0.,1.]])]
DEFINITIONS=['quality_ge_6','quality_ge_7']
def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);r=np.empty_like(p);run=0.;m=len(p)
 for k,i in enumerate(o): run=max(run,(m-k)*p[i]);r[i]=min(run,1.)
 return r
def crt(d,ratio=.25):
 d=np.asarray(d,float);n=len(d);mean=d.mean();var=d.var(ddof=1);se=np.sqrt((1/n+ratio)*var);t=mean/se if se else (0 if mean==0 else np.sign(mean)*np.inf);p=2*student_t.sf(abs(t),df=n-1) if np.isfinite(t) else 0.;crit=student_t.ppf(.975,df=n-1);return float(mean),float(se),float(t),float(p),float(mean-crit*se),float(mean+crit*se)
def fit_quantile_bins(X,n_bins=5):
 cuts=[]
 for j in range(X.shape[1]): cuts.append(np.unique(np.quantile(X[:,j],np.linspace(0,1,n_bins+1)[1:-1])))
 return cuts
def apply_bins(X,cuts): return np.column_stack([np.digitize(X[:,j],cuts[j],right=False) for j in range(X.shape[1])]).astype(int)
def run(datafile:Path,out:Path,seed=20260821,n_repeats=5):
 out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(datafile,sep=';'); feature_cols=[c for c in df.columns if c!='quality']; X=df[feature_cols].to_numpy(float); q=df.quality.to_numpy(int); y=np.where(q>=7,2,np.where(q==6,1,0)).astype(int); obs=np.column_stack([(q>=6).astype(int),(q>=7).astype(int)]); groups=pd.util.hash_pandas_object(df[feature_cols],index=False).to_numpy(np.uint64); rank=coarsening_rank(OPERATORS); assert rank['full_column_rank']
 rows=[];aud=[];preds=[]
 for rep in range(1,n_repeats+1):
  rs=seed+(rep-1)*10007;cv=StratifiedGroupKFold(5,shuffle=True,random_state=rs)
  for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
   split=f'R{rep}F{fold}'; cuts=fit_quantile_bins(X[tr],5); Xdtr=apply_bins(X[tr],cuts); Xdte=apply_bins(X[te],cuts); rng=np.random.default_rng(rs+1000+fold);env=random_environment_split(len(tr),rng)%2; xdenv=[Xdtr[env==j] for j in range(2)];yoenv=[obs[tr][env==j,j] for j in range(2)]
   prop=GaliatsatosMethod(n_classes=3,structure='tan',smoothing=.10,max_iter=120,tol=1e-5,n_init=3,init_jitter=.05,random_state=rs+fold).fit(xdenv,yoenv,OPERATORS);pp=prop.predict_proba(Xdte)
   sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te]);xenv=[Xtr[env==j] for j in range(2)];opl=OperatorAwareLogisticEM(3,80,1e-6,1.).fit(xenv,yoenv,OPERATORS);po=opl.predict_proba(Xte);oracle=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=rs+fold).fit(Xtr,y[tr]);raw=oracle.predict_proba(Xte);por=np.full((len(te),3),1e-12)
   for cp,c in enumerate(oracle.classes_.astype(int)):por[:,c]=raw[:,cp]
   por/=por.sum(1,keepdims=True);probs={METHODS[0]:pp,METHODS[1]:po,METHODS[2]:por}
   for method,p in probs.items():
    rows.append(dict(repeat=rep,fold=fold,split_id=split,evaluation='canonical_quality',method=method,**metrics(y[te],p)))
    for j,name in enumerate(DEFINITIONS):
     pdx=p@OPERATORS[j].T;pdx/=pdx.sum(1,keepdims=True);rows.append(dict(repeat=rep,fold=fold,split_id=split,evaluation='transport_'+name,method=method,**metrics(obs[te,j],pdx)))
    pr=p.argmax(1)
    for k,idx in enumerate(te):preds.append(dict(row_id=int(idx),repeat=rep,fold=fold,split_id=split,method=method,true=int(y[idx]),pred=int(pr[k]),p0=float(p[k,0]),p1=float(p[k,1]),p2=float(p[k,2])))
   aud.append(dict(repeat=rep,fold=fold,split_id=split,n_train=len(tr),n_test=len(te),n_train_groups=int(pd.Series(groups[tr]).nunique()),n_test_groups=int(pd.Series(groups[te]).nunique()),group_overlap=int(len(set(groups[tr]).intersection(set(groups[te])))),environment_sizes=json.dumps([int((env==j).sum()) for j in range(2)]),operator_rank=rank['rank'],operator_condition_number=rank['condition_number_on_identifiable_subspace'],converged=prop.converged_,termination_reason=prop.termination_reason_,iterations=prop.n_iter_,best_start=prop.best_start_,canonical_labels_used=False))
   print(f'{split}: n={len(te)} PLA acc={metrics(y[te],pp)["accuracy"]:.3f} macroF1={metrics(y[te],pp)["macro_f1"]:.3f} term={prop.termination_reason_}',flush=True)
 m=pd.DataFrame(rows);m.to_csv(out/'wine_metrics_by_fold.csv',index=False);pd.DataFrame(aud).to_csv(out/'wine_fold_audit.csv',index=False);pd.DataFrame(preds).to_csv(out/'wine_oof_predictions.csv',index=False);can=m[m.evaluation=='canonical_quality'];summ=can.groupby('method',as_index=False).agg(n_outer=('accuracy','size'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),log_loss_mean=('log_loss','mean'),log_loss_sd=('log_loss','std'),brier_mean=('brier','mean'),brier_sd=('brier','std'));summ.to_csv(out/'wine_canonical_summary.csv',index=False);repm=can.groupby(['repeat','method'],as_index=False)[METRICS].mean();repm.to_csv(out/'wine_repeat_means.csv',index=False)
 tests=[]
 for comp in METHODS[1:]:
  for met in METRICS:
   a=can[can.method==METHODS[0]].set_index('split_id')[met];b=can[can.method==comp].set_index('split_id')[met];mean,se,t,p,lo,hi=crt((a-b).to_numpy());ar=repm[repm.method==METHODS[0]].set_index('repeat')[met];br=repm[repm.method==comp].set_index('repeat')[met];w=wilcoxon((ar-br).to_numpy(),method='exact');tests.append(dict(comparison=f'{METHODS[0]} minus {comp}',metric=met,mean_difference=mean,corrected_se=se,corrected_t=t,corrected_p=p,ci95_low=lo,ci95_high=hi,wilcoxon_p=float(w.pvalue)))
 tdf=pd.DataFrame(tests);tdf['corrected_p_holm']=holm(tdf.corrected_p);tdf['wilcoxon_p_holm']=holm(tdf.wilcoxon_p);tdf.to_csv(out/'wine_paired_tests.csv',index=False);m[m.evaluation.str.startswith('transport_')].groupby(['evaluation','method'],as_index=False).agg(accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),macro_f1_mean=('macro_f1','mean'),log_loss_mean=('log_loss','mean'),brier_mean=('brier','mean')).to_csv(out/'wine_transport_summary.csv',index=False)
 ad=pd.DataFrame(aud);manifest={'dataset':'UCI Wine Quality white','n_rows':len(df),'n_unique_predictor_vectors':int(pd.Series(groups).nunique()),'canonical_counts':{str(k):int(v) for k,v in pd.Series(y).value_counts().sort_index().items()},'quality_score_counts':{str(k):int(v) for k,v in df.quality.value_counts().sort_index().items()},'definition_positive_rates':{'quality_ge_6':float(obs[:,0].mean()),'quality_ge_7':float(obs[:,1].mean())},'operator_rank':rank['rank'],'operator_condition_number':rank['condition_number_on_identifiable_subspace'],'all_group_overlap_zero':bool((ad.group_overlap==0).all()),'all_tolerance_converged':bool((ad.termination_reason=='tolerance').all()),'n_repeats':n_repeats,'total_outer':5*n_repeats,'binning':'5 training-fold quantile bins per feature'};(out/'wine_manifest.json').write_text(json.dumps(manifest,indent=2));print('\nSUMMARY\n',summ.to_string(index=False));print('\nMANIFEST\n',json.dumps(manifest,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-file',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--seed',type=int,default=20260821);ap.add_argument('--n-repeats',type=int,default=5);a=ap.parse_args();run(a.data_file,a.output_dir,a.seed,a.n_repeats)
