import pandas as pd
from tqdm.auto import tqdm

def get_center_ord1_data(url):
    data = pd.read_csv(url)
    new_url = url.replace(".csv","_center_ord1.csv")
    new_data = data.loc[data['name'].str.contains('center_ord1')]
    new_data.to_csv(new_url,index = False)

# def get_three_view_ord1_data(url):
#     missing = []
#     data = pd.read_csv(url)
#     new_url = url.replace(".csv","_three_view_ord1.csv")
#     full = []
#     center = data.loc[data['name'].str.contains('center_ord1')]
#     for idx,row in tqdm(center.iterrows()):
#         center_name = row['name']
#         label = row['label']
#         # head = center_name.split('___')[0]
#         head = '_'.join(center_name.split('_')[:2])

#         tail = '_'+'_'.join(center_name.split('_')[-2:])
#         if 'ord1' in tail:
#             result = data[data['name'].str.startswith(head) & data['name'].str.contains(tail)]

#             names = result.values
            
#             if len(names) == 3:
#                 center_name = None
#                 right_name = None
#                 left_name = None
#                 assert names[0][1] == label
#                 assert names[1][1] == label
#                 assert names[2][1] == label
               
#                 for name in names:
#                     name = name[0]
#                     if 'center' in name:
#                         if center_name is not None:
#                             print(head,tail,names)
#                         assert center_name is None
#                         center_name = name
#                     if 'right' in name:
#                         if right_name is not None:
#                             print(head,tail,names)
#                         assert right_name is None
#                         right_name = name
#                     if 'left' in name:
#                         if left_name is not None:
#                             print(head,tail,names)
#                         assert left_name is None
#                         left_name = name
                
#                 assert center_name is not None
#                 assert right_name is not None
#                 assert left_name is not None
              
#                 full.append([center_name,left_name,right_name,label])
#             elif len(names) < 3:
#                 m = []
#                 for name in names:
#                     m.append(name[0])
#                 m += (3-len(m))*['missing']
#                 missing.append(m)
#             else:
#                 print(len(names))
   
#     new_data = pd.DataFrame(missing,columns = ['1','2','3'])
#     new_data.to_csv('missing.csv',index = False)
   
#     new_data = pd.DataFrame(full,columns = ['center','left','right','label'])
#     new_data.to_csv(new_url,index = False)
#     # new_data.to_csv('demo',index = False)
    
def get_three_view_ord1_data(url,url1):
    missing = []
    data = pd.read_csv(url)
    new_url = url.replace(".csv","_three_view_ord1.csv")
    full = []
    center = pd.read_csv(url1)
    for idx,row in tqdm(center.iterrows()):
        center_name = row['name']
        label = row['label']
        head = '_'.join(center_name.split('_')[:2])
        middle = center_name.split('___')[0].split('_')[-1:]
        tail = '_'+'_'.join(center_name.split('_')[-2:])
        cls_ = center_name.split("___")[0]
        if 'ord1' in tail:
            result = data[data['name'].str.startswith(head) & data['name'].str.contains(tail) ]
            names = result.values
            if len(names) > 3:
                result = data[data['name'].str.startswith(cls_) & data['name'].str.contains(tail)]
                names = result.values
            
            if len(names) == 3:
                center_name = None
                right_name = None
                left_name = None
                assert names[0][1] == label
                assert names[1][1] == label
                assert names[2][1] == label
               
                for name in names:
                    name = name[0]
                    if 'center' in name:
                        if center_name is not None:
                            print(head,tail,names)
                        assert center_name is None
                        center_name = name
                    if 'right' in name:
                        if right_name is not None:
                            print(head,tail,names)
                        assert right_name is None
                        right_name = name
                    if 'left' in name:
                        if left_name is not None:
                            print(head,tail,names)
                        assert left_name is None
                        left_name = name
                
                assert center_name is not None
                assert right_name is not None
                assert left_name is not None
              
                full.append([center_name,left_name,right_name,label])
            elif len(names) < 3:
                m = []
                for name in names:
                    m.append(name[0])
                m += (3-len(m))*['missing']
                missing.append(m)
   
    new_data = pd.DataFrame(missing,columns = ['1','2','3'])
    new_data.to_csv('missing.csv',index = False)
   
    new_data = pd.DataFrame(full,columns = ['center','left','right','label'])
    new_data.to_csv(new_url,index = False)
    # new_data.to_csv('demo',index = False)

if __name__ == "__main__":
    # get_center_ord1_data('train_1_200.csv')
    # get_center_ord1_data('val_1_200.csv')
    # get_center_ord1_data('test_1_200.csv')
    # get_center_ord1_data('full_data_1_200.csv')

    # get_three_view_ord1_data('train_1_200.csv','train_1_200_center_ord1.csv')
    # get_three_view_ord1_data('test_1_200.csv','test_1_200_center_ord1.csv')
    # get_three_view_ord1_data('val_1_200.csv','val_1_200_center_ord1.csv')
    get_three_view_ord1_data('full_data_1_200.csv','full_data_1_200_center_ord1.csv')