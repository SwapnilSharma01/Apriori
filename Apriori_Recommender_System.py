# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pip install apyori
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv("E:/Swapnil/Data_Science/Machine_Learning/Apriori_Algorithm//Dataset.csv")


# Grouping the data based on 'Time_of_Change' and concatnating Changed_Entity_ID
flatten_col_df = dataset.groupby(['Time_of_Change'])['Changed_Entity_ID'].apply(lambda x: ','.join(x)).reset_index()

#flatten_col_df_1 = dataset.groupby(['Time_of_Change'])['Changed_Entity_ID'].apply(lambda x: ' '.join(x)).reset_index()

flatten_df = flatten_col_df.Changed_Entity_ID.str.split(',', expand=True)

#flatten_list =  flatten_col_df['Changed_Entity_ID'].tolist()

row_count = flatten_col_df['Changed_Entity_ID'].count()

column_count = len(flatten_df.columns)

        
item_list_of_list = []
for rows in range(0, row_count):
    item_list_of_list.append([str(flatten_df.values[rows,cols]) for cols in range(0, column_count)])

"""
#pip install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(item_list_of_list, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

#frequent_itemsets_rules = apriori(item_list_of_list, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

frequent_itemsets = apriori(item_list_of_list, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

print(list(rules))
"""

frequent_itemsets_rules = apriori(item_list_of_list, min_support = 0.4, min_confidence = 0.2, min_lift = 2, min_length = 2)


rules_df = pd.DataFrame(frequent_itemsets_rules)

print(list(frequent_itemsets_rules))

