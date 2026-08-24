from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from nhanes_diabetes_validation import discrete_features, continuous_preprocessor, metrics, random_environment_split
from plabn import GaliatsatosMethod, coarsening_rank
from publication_experiments import OperatorAwareLogisticEM

K=3
METHODS=["GALIATSATOS/PLA-BN TAN","Operator-aware logistic EM","Oracle-label logistic"]
METRICS=["accuracy","balanced_accuracy","macro_f1","log_loss","brier"]
OP130=np.array([[1.,0.,0.],[0.,1.,1.]])
OP140=np.array([[1.,1.,0.],[0.,0.,1.]])
OPERATORS=[OP130,OP140]
DEFINITIONS=["ACC_AHA_130_80","ESC_140_90"]

def holm_adjust(pvalues):
    pvalues=np.asarray(pvalues,float); order=np.argsort(pvalues); out=np.empty_like(pvalues); running=0.; m=len(pvalues)
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*pvalues[idx]); out[idx]=min(running,1.)
    return out

def corrected_resampled_t(diff, ratio=.25):
    diff=np.asarray(diff,float); n=len(diff); mean=float(diff.mean()); var=float(diff.var(ddof=1)); se=float(np.sqrt((1/n+ratio)*var))
    if se==0: tstat=0. if mean==0 else np.sign(mean)*np.inf; p=1. if mean==0 else 0.; lo=hi=mean
    else:
        tstat=mean/se; p=float(2*student_t.sf(abs(tstat),df=n-1)); crit=float(student_t.ppf(.975,df=n-1)); lo=mean-crit*se; hi=mean+crit*se
    return mean,se,float(tstat),p,float(lo),float(hi)

def load_data(data_dir: Path):
    demo=pd.read_csv(data_dir/'DEMO_I.csv')
    bmx=pd.read_csv(data_dir/'BMX_I.csv')
    bpx=pd.read_csv(data_dir/'BPX_I.csv')
    frame=(demo[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH3","INDFMPIR","DMDEDUC2"]]
           .merge(bmx[["SEQN","BMXBMI","BMXWAIST"]],on='SEQN',how='inner')
           .merge(bpx[["SEQN"]+[f"BPXSY{i}" for i in range(1,5)]+[f"BPXDI{i}" for i in range(1,5)]],on='SEQN',how='inner'))
    frame=frame.loc[frame.RIDAGEYR>=20].copy()
    sy=frame[[f"BPXSY{i}" for i in range(1,5)]].to_numpy(float)
    di=frame[[f"BPXDI{i}" for i in range(1,5)]].to_numpy(float)
    valid=np.isfinite(sy)&np.isfinite(di)&(sy>=50)&(sy<=300)&(di>=0)&(di<=200)&(sy>di)
    nvalid=valid.sum(axis=1)
    sy2=np.where(valid,sy,np.nan); di2=np.where(valid,di,np.nan)
    frame['mean_sbp']=np.nanmean(sy2,axis=1); frame['mean_dbp']=np.nanmean(di2,axis=1); frame['n_bp_pairs']=nvalid
    frame=frame.loc[(frame.n_bp_pairs>=2)&frame.BMXBMI.notna()&frame.BMXWAIST.notna()&frame.RIAGENDR.notna()&frame.RIDRETH3.notna()].reset_index(drop=True)
    high130=(frame.mean_sbp>=130)|(frame.mean_dbp>=80)
    high140=(frame.mean_sbp>=140)|(frame.mean_dbp>=90)
    frame['canonical_bp']=np.where(high140,2,np.where(high130,1,0)).astype(int)
    frame['def130']=high130.astype(int); frame['def140']=high140.astype(int)
    return frame

def run(data_dir:Path, out:Path, seed=20260821, n_repeats=5):
    out.mkdir(parents=True,exist_ok=True); frame=load_data(data_dir)
    y=frame.canonical_bp.to_numpy(int); observed=np.column_stack([frame.def130,frame.def140]).astype(int)
    X_disc=discrete_features(frame)
    predcols=["RIDAGEYR","BMXBMI","BMXWAIST","INDFMPIR","RIAGENDR","RIDRETH3","DMDEDUC2"]
    X_raw=frame[predcols].copy()
    rank=coarsening_rank(OPERATORS)
    assert rank['full_column_rank']
    rows=[]; audits=[]; preds=[]
    for rep in range(1,n_repeats+1):
        rs=seed+(rep-1)*10007; cv=StratifiedKFold(5,shuffle=True,random_state=rs)
        for fold,(tr,te) in enumerate(cv.split(X_raw,y),1):
            split=f"R{rep}F{fold}"; rng=np.random.default_rng(rs+1000+fold); env=random_environment_split(len(tr),rng)%2
            xdenv=[X_disc[tr][env==j] for j in range(2)]; yoenv=[observed[tr][env==j,j] for j in range(2)]
            prop=GaliatsatosMethod(n_classes=3,structure='tan',smoothing=.10,max_iter=120,tol=1e-5,n_init=3,init_jitter=.05,random_state=rs+fold).fit(xdenv,yoenv,OPERATORS)
            pp=prop.predict_proba(X_disc[te])
            prep=continuous_preprocessor(); Xtr=prep.fit_transform(X_raw.iloc[tr]); Xte=prep.transform(X_raw.iloc[te]); xenv=[Xtr[env==j] for j in range(2)]
            opl=OperatorAwareLogisticEM(n_classes=3,max_iter=80,tol=1e-6,c_value=1.).fit(xenv,yoenv,OPERATORS); po=opl.predict_proba(Xte)
            oracle=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=rs+fold).fit(Xtr,y[tr]); raw=oracle.predict_proba(Xte); por=np.full((len(te),3),1e-12)
            for cpos,c in enumerate(oracle.classes_.astype(int)): por[:,c]=raw[:,c]
            por/=por.sum(axis=1,keepdims=True)
            probs={METHODS[0]:pp,METHODS[1]:po,METHODS[2]:por}
            for method,p in probs.items():
                rows.append(dict(repeat=rep,fold=fold,split_id=split,evaluation='canonical_bp_severity',method=method,**metrics(y[te],p)))
                for j,name in enumerate(DEFINITIONS):
                    pdx=p@OPERATORS[j].T; pdx/=pdx.sum(axis=1,keepdims=True)
                    rows.append(dict(repeat=rep,fold=fold,split_id=split,evaluation='transport_'+name,method=method,**metrics(observed[te,j],pdx)))
                pred=p.argmax(1)
                for k,idx in enumerate(te): preds.append(dict(SEQN=int(frame.iloc[idx].SEQN),repeat=rep,fold=fold,split_id=split,method=method,true=int(y[idx]),pred=int(pred[k]),p0=float(p[k,0]),p1=float(p[k,1]),p2=float(p[k,2])))
            audits.append(dict(repeat=rep,fold=fold,split_id=split,n_train=len(tr),n_test=len(te),environment_sizes=json.dumps([int((env==j).sum()) for j in range(2)]),operator_rank=rank['rank'],operator_condition_number=rank['condition_number_on_identifiable_subspace'],proposed_converged=prop.converged_,termination_reason=prop.termination_reason_,iterations=prop.n_iter_,best_start=prop.best_start_,canonical_labels_used_for_proposed_fit=False,outer_test_used_for_preprocessing=False))
            print(f"{split}: n={len(te)} PLA acc={metrics(y[te],pp)['accuracy']:.3f} macroF1={metrics(y[te],pp)['macro_f1']:.3f} term={prop.termination_reason_}",flush=True)
    m=pd.DataFrame(rows); m.to_csv(out/'hypertension_metrics_by_fold.csv',index=False); pd.DataFrame(audits).to_csv(out/'hypertension_fold_audit.csv',index=False); pd.DataFrame(preds).to_csv(out/'hypertension_oof_predictions.csv',index=False)
    can=m[m.evaluation=='canonical_bp_severity'].copy(); summ=can.groupby('method',as_index=False).agg(n_outer=('accuracy','size'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),log_loss_mean=('log_loss','mean'),log_loss_sd=('log_loss','std'),brier_mean=('brier','mean'),brier_sd=('brier','std')); summ.to_csv(out/'hypertension_canonical_summary.csv',index=False)
    repmeans=can.groupby(['repeat','method'],as_index=False)[METRICS].mean(); repmeans.to_csv(out/'hypertension_repeat_means.csv',index=False)
    tests=[]
    for comp in METHODS[1:]:
        for met in METRICS:
            a=can[can.method==METHODS[0]].set_index('split_id')[met]; b=can[can.method==comp].set_index('split_id')[met]; d=(a-b).to_numpy(); mean,se,t,p,lo,hi=corrected_resampled_t(d)
            ar=repmeans[repmeans.method==METHODS[0]].set_index('repeat')[met]; br=repmeans[repmeans.method==comp].set_index('repeat')[met]; dr=(ar-br).to_numpy(); w=wilcoxon(dr,method='exact')
            tests.append(dict(comparison=f"{METHODS[0]} minus {comp}",metric=met,mean_difference=mean,corrected_se=se,corrected_t=t,corrected_p=p,ci95_low=lo,ci95_high=hi,wilcoxon_p=float(w.pvalue)))
    tdf=pd.DataFrame(tests); tdf['corrected_p_holm']=holm_adjust(tdf.corrected_p); tdf['wilcoxon_p_holm']=holm_adjust(tdf.wilcoxon_p); tdf.to_csv(out/'hypertension_paired_tests.csv',index=False)
    trans=m[m.evaluation.str.startswith('transport_')].groupby(['evaluation','method'],as_index=False).agg(accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),macro_f1_mean=('macro_f1','mean'),log_loss_mean=('log_loss','mean'),brier_mean=('brier','mean')); trans.to_csv(out/'hypertension_transport_summary.csv',index=False)
    manifest={'dataset':'NHANES 2015-2016','n_participants':len(frame),'bp_pair_requirement':'>=2 valid paired readings','mean_sbp':float(frame.mean_sbp.mean()),'mean_dbp':float(frame.mean_dbp.mean()),'canonical_counts':{str(k):int(v) for k,v in frame.canonical_bp.value_counts().sort_index().items()},'definition_positive_rates':{'130_80':float(frame.def130.mean()),'140_90':float(frame.def140.mean())},'operator_rank':rank['rank'],'operator_condition_number':rank['condition_number_on_identifiable_subspace'],'all_tolerance_converged':bool((pd.DataFrame(audits).termination_reason=='tolerance').all()),'n_repeats':n_repeats,'total_outer':5*n_repeats}
    (out/'hypertension_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUMMARY\n',summ.to_string(index=False)); print('\nMANIFEST\n',json.dumps(manifest,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260821); ap.add_argument('--n-repeats',type=int,default=5); a=ap.parse_args(); run(a.data_dir,a.output_dir,a.seed,a.n_repeats)
