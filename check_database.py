from torch.utils import data
from semantic_seg_v4 import SegmentationDataSet_3legs



liste_data = []
#dict_nb_layer = {'228': 553, '234': 415, '239': 873, '240': 873, '269': 373, '421': 423, '428': 323, '437': 198, '474': 748, '481': 467, '488': 1182, '496': 91, '507': 317, '514': 1840}
dict_nb_layer = {'496': 91, '507': 317, '514': 1840}

#dict_nb_layer = {'238': 998}
non_valid = [['213','00190'],['213','00277'],['213','00301'],['213','00388'],['225','00629'],['225','00468'],['225','00485'],['225','00550'],['225','00558'],['225','00614'],['225','00632'],['225','00638'],['238','00398'],['238','00586'],['238','00599'],['238','00759'],['238','00928'],['265','00467'],['265','00661'],['488','00999']]
#numero_build=['228','269','428']
#for i in range(batch_size*nb_mini_batch):
for key in dict_nb_layer.keys():
    nb_build=key
    #nb_layer=np.linspace(17,dict_nb_layer[key],dict_nb_layer[key]-17)
    for nb_layer in range(17,dict_nb_layer[key]):
        for nb_tile in range(25):
        #for nb_tile in [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]: #without the corner tile
            if nb_layer<10:
                    layer=f'0000{nb_layer}'
            if nb_layer<100:
                    layer=f'000{nb_layer}'
            if 99<nb_layer<1000:
                    layer=f'00{nb_layer}'
            if 999<nb_layer<10000:
                    layer=f'0{nb_layer}'
            if [nb_build,layer] not in non_valid :
                    liste_data.append([nb_build,layer,str(nb_tile)])

dataset=SegmentationDataSet_3legs(liste_data)

dataloader = data.DataLoader(dataset=dataset,
                                        batch_size=1,
                                        shuffle=False)

for batch_idx, (A, B, D, labels) in enumerate(dataloader):
    print(f'type{liste_data[batch_idx]}')


#213 okay
#225 pas le bon stl useless