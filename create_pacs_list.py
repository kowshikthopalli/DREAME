import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

root = 'DATA/PACS'
domains = sorted([f.name for f in os.scandir(root) if f.is_dir()])


class_names = sorted([f.name for f in os.scandir(os.path.join(root,domains[0])) if f.is_dir()])
le = preprocessing.LabelEncoder()
le.fit(class_names)

def create_list_df(directory,class_names):
    dfs_all=[]
    for class_name in class_names:
        path = os.path.join(directory,class_name)
        files_this_class= sorted([os.path.join(path,f.name) for f in os.scandir(path) ])
        labels = [le.transform([class_name])[0]]*len(files_this_class)
        df= pd.DataFrame([files_this_class,labels]).T
        df.columns = ['paths','labels']
        dfs_all.append(df)
    dfs_all = pd.concat(dfs_all)
    return dfs_all

"""
To dump original images in that order as CSV with headings paths and labels
"""
domain_dfs=[]
for domain in domains:
    directory= os.path.join(root,domain)

    domain_df= create_list_df(directory,class_names)
    #domain_df.to_csv('PACS_original_list_files/'+domain+'.csv', index=False)

    domain_dfs.append(domain_df)
complete_data_df = pd.concat(domain_dfs)
"""
Now have to split this into five different domains. domain_1.csv, domain_2.csv etc randomly
"""
for seed in [102,334,234,12,987]:

    np.random.seed(seed)
    shuffled_df = complete_data_df.sample(frac=1,random_state=seed)

    result_df = np.array_split(shuffled_df, 5) 

    save_dir = 'PACS_splits/seed_'+str(seed)
    os.makedirs(save_dir, exist_ok=True)
    for i,result in enumerate(result_df):
        save_file= os.path.join(save_dir,'domain_'+str(i)+'.csv')
        result.to_csv(save_file, index=False)

