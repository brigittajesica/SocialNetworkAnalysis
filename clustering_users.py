import pandas as pd
import os

num = 0
i = 0
j = 0

def cleanLabel(text):
    if text.endswith('&#10;'):
        text = text[:-5]
    else:
        pass
    return text

clustered_df = pd.read_csv('clustered_df.csv')
os.chdir('Following Lists')
table_df = pd.read_csv('table.csv')

table_df['Label'] = table_df['Label'].apply(cleanLabel)

for file_name in os.listdir():
    if i == 10:
        # table_df.to_csv('final_df' + str(j) + '.csv', index = False)
        # print(f'FINAL CSV NUMBER {j+1} created')
        print(f"Step: {j+1}")
        i = 0
        j += 1
    else:
        if file_name.startswith('followinglist'):
            i += 1
            print(file_name)
            if len(file_name)==18:
                num = int(file_name[-5:-4])
            elif len(file_name)==19:
                num = int(file_name[-6:-4])
            elif len(file_name)==20:
                num = int(file_name[-7:-4])
            cluster = clustered_df['Cluster'].loc[num]
            try:
                temp_df = pd.read_csv(file_name, header=None)
                for user in temp_df[0]:
                    table_df.loc[table_df['Label'] == user, 'Cluster'] = cluster
            except:
                pass

table_df.to_csv('final_df.csv', index = False)