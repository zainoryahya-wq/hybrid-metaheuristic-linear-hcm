import pandas as pd, numpy as np
s=pd.read_csv('results/sensitivity_raw.csv'); raw=pd.read_csv('results/raw_results.csv')
raw=raw[raw.algo!='Standard'].copy(); raw['eps']=1e-2
pd.set_option('display.width',250)
g=s.groupby(['system','algo','eps']).agg(succ=('success','mean'),nmeta=('n_meta','mean'),conv=('meta_converged','mean'),nhcm=('n_hcm','mean'),cpu=('cpu_total','mean')).reset_index()
g['succ']*=100; g['conv']*=100
print(g.pivot_table(index=['system','algo'],columns='eps',values=['succ','conv','nmeta']).round(1).to_string())
# agreement of eps=1e-2 rerun with raw
m=s[s.eps==1e-2].merge(raw,on=['system','algo','seed'],suffixes=('_r','_o'))
print('rows',len(m),'n_meta equal',(m.n_meta_r==m.n_meta_o).mean(),'success equal',(m.success_r==m.success_o).mean(),'n_hcm equal',(m.n_hcm_r==m.n_hcm_o).mean())
print(m.groupby(['system','algo']).apply(lambda d:pd.Series(dict(succ_r=d.success_r.mean()*100,succ_o=d.success_o.mean()*100,nm_eq=(d.n_meta_r==d.n_meta_o).mean()))).round(2).to_string())
