# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:24:43 2021

@author: zishi
"""

import pandas as pd
from sklearn import metrics

def ROC(df_label, pred, comb_name):
    comb_auc =  metrics.roc_auc_score(df_label,pred)
    comb_pred = (pred > 0.5)
    comb_acc = metrics.accuracy_score(df_label, comb_pred)
    print(comb_name+' ensemble ROC_AUC: %0.4f' %comb_auc)
    print(comb_name+' ensemble ACC: %0.4f' %comb_acc) 
    
def ensemble(df_list, mode='hard'):
    
    df = pd.concat(df_list)
    df_mean = df.groupby(df.index).mean()
    
    if mode == 'hard':
        return df_mean['label']
        
    else:
        return df_mean['proba']


df = pd.read_json(r'C:\Users\zishi\Downloads\data\dev_unseen.jsonl',lines=True)
df_label = df[['id','label']].copy()
df_label.set_index('id',inplace=True)
df_label.sort_index(inplace=True)

df_vilbert_cc = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\VilBERT on CC.csv',index_col='id')
df_visualbert_coco = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\VisualBERT on COCO.csv',index_col='id')
df_concatbert = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\ConcatBERT.csv',index_col='id')
df_latefusion = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\Late Fusion.csv',index_col='id')
df_mmbtgrid = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\MMBT-Grid.csv',index_col='id')
df_mmbtregion = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\MMBT-Region.csv',index_col='id')
df_vilbert = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\VilBERT.csv',index_col='id')
df_visualbert = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Unimodal Trained\Visual Bert.csv',index_col='id')
df_lxmert = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\LXMERT\LXMERT pretrained on multiple.csv',index_col='id')

df_Gu = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\GuYi\Gu_val.csv',index_col='id')

df_img_large_4l = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\img_large_4l_val.csv',index_col='id')
df_img_large_2l = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\img_large_2l_val.csv',index_col='id')
df_text_large_4l = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\text_large_4l_val.csv',index_col='id')
df_text_large_2l = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\text_large_2l_val.csv',index_col='id')
df_VB = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\VB_val.csv',index_col='id')
df_VB_large = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\VB_large_val.csv',index_col='id')
df_VBCOCO = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\VBCOCO_val.csv',index_col='id')
df_VBCOCO_large = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\WenLei\VBCOCO_val.csv',index_col='id')

df_roberta = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Roberta\roberta_dropout1_val.csv',index_col='id')

# Named List for cominations
pre_multi = [df_vilbert_cc,df_visualbert_coco]
pre_uni = [df_concatbert,df_latefusion,df_mmbtgrid,df_mmbtregion,df_vilbert,df_visualbert]
pre_combine = pre_multi + pre_uni
pre_combineLX = pre_combine + [df_lxmert]

uni_2l = [df_img_large_2l,df_text_large_2l]
uni_4l = [df_img_large_4l,df_text_large_4l]
uni = uni_2l + uni_4l

uni_total = uni_2l + uni_4l + pre_uni
VB = [df_VBCOCO_large,df_VB_large]

#Combinations
cur = [df_Gu] + pre_uni
ens_h = ensemble(cur,'hard')
ens_s = ensemble(cur,'soft')
ROC(df_label,ens_h,'Gu+pre_uni hard')
ROC(df_label,ens_s,'Gu+pre_uni soft')

cur_can1 = [df_Gu,df_VBCOCO_large] 
cur_can2 = [df_Gu] + pre_multi









