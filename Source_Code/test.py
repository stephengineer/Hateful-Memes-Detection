# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:33:19 2021

@author: zishi
"""

import pandas as pd

def ensemble_t(df_list,mode='soft'):
    df = pd.concat(df_list)
    df_mean = df.groupby(df.index).mean()
    if mode == 'soft':
        result = pd.DataFrame(df_mean['proba'],index=df_list[0].index)
    else:
        result = pd.DataFrame(df_mean['label'],index=df_list[0].index)
        result = result.rename(columns={'label':'proba'})
    result['label'] = (result['proba'] > 0.5) * 1
    
    return result
    

df_vilbert_cc = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\ViLBCC_pretest.csv',index_col='id')
df_visualbert_coco = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\VBCOCO_pretest.csv',index_col='id')
df_Gu = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\Gu_test.csv',index_col='id')
df_VBCOCO_large = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\VBCOCO_testl.csv',index_col='id')
df_VB_large = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\VB_testl.csv',index_col='id')

df_roberta = pd.read_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\roberta_dropout1_test.csv',index_col='id')

#Named list
df12 = [df_Gu,df_VBCOCO_large,df_VB_large]
df2 = [df_Gu,df_vilbert_cc,df_visualbert_coco]
df13 = [df_Gu, df_VBCOCO_large,df_VB_large,df_vilbert_cc,df_visualbert_coco]
df4 = [df_Gu,df_VBCOCO_large]
df6 = [df_Gu,df_vilbert_cc,df_visualbert_coco]
df11 = [df_Gu, df_VB_large]
df15 = [df_Gu, df_roberta]
df16 = [df_Gu,df_roberta,df_VBCOCO_large,df_VB_large]
df17 = [df_Gu,df_roberta,df_VBCOCO_large,df_VB_large,df_vilbert_cc,df_visualbert_coco]

df_tests12 = ensemble_t(df12)
df_testh12 = ensemble_t(df12,'hard')
df_tests2 = ensemble_t(df2)
df_tests13 = ensemble_t(df13)
df_testh13 = ensemble_t(df13,'hard')
df_tests4 = ensemble_t(df4)
df_tests11 = ensemble_t(df11)
df_testh11 = ensemble_t(df11,'hard')
df_tests15 = ensemble_t(df15)
df_tests16 = ensemble_t(df16)
df_testh16 = ensemble_t(df16,'hard')
df_tests17 = ensemble_t(df17)
df_testh17 = ensemble_t(df17,'hard')
df_tests6 = ensemble_t(df6)

df_tests2.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests2.csv')
df_tests12.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests12.csv')
df_tests12.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\testh12.csv')
df_tests13.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests13.csv')
df_testh13.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\testh13.csv')
df_tests4.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests4.csv')
df_tests11.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests11.csv')
df_testh11.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\testh11.csv')

df_tests15.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests15.csv')
df_tests16.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests16.csv')
df_testh16.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\testh16.csv')
df_tests17.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests17.csv')
df_testh17.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\testh17.csv')
df_tests6.to_csv(r'C:\Users\zishi\DL\Group Project\Inference\Test\tests6.csv')
     