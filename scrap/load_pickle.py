import pickle
import random
import os
ubpmc_data_partition = None
adobe_synth = None

with open('/home/danny/Downloads/split4_train_task6_all.pkl', 'rb') as f:
    ubpmc_data = pickle.load(f)

    sdf = 0

ubpmc_data_partition = dict()

for val in ubpmc_data:
    random.shuffle(ubpmc_data[val])
    all_data = []
    shuffled = ubpmc_data[val]
    for gr in range(len(shuffled)):
        list_t=[0]*2
        list_t[0] = f"data/ubpmc/annotations_JSON/{val}/{shuffled[gr]}"
        list_t[1] = f"data/ubpmc/images/{val}/{os.path.splitext(os.path.basename(shuffled[gr]))[0]}.jpg"
        all_data.append(list_t)
    shuffled = all_data
    part = len(shuffled) // 4
    diff = len(shuffled) - part*4
    
    part1 = shuffled[0*part:part]
    part2 = shuffled[1*part:2*part]
    part3 = shuffled[2*part:3*part]
    part4 = shuffled[3*part:4*part+diff]
    ubpmc_data_partition[val] = []

    ubpmc_data_partition[val].append(part1)
    ubpmc_data_partition[val].append(part2)
    ubpmc_data_partition[val].append(part3)
    ubpmc_data_partition[val].append(part4) 
    
    assert len(shuffled) == (len(part1)+len(part2)+len(part3)+len(part4))


with open('/home/danny/Downloads/train_task6_syntyPMC.pkl', 'rb') as f:
    adobe_synth = pickle.load(f)
    
    

    sdf = 0
adobe_data_partition = dict()
for val in adobe_synth:
    random.shuffle(adobe_synth[val])
    shuffled = adobe_synth[val]
    if val == "line2":
        val = 'line'
    if val == 'scatter2':
        val = 'scatter'
    all_data = []
    for gr in range(len(shuffled)):
        list_t=[0]*2
        
        list_t[0] = f"data/adobesynth/Task-level-JSONs/JSONs/{val}/{shuffled[gr]}"
        list_t[1] = f"data/adobesynth/Chart-Images-and-Metadata/ICPR/Charts/{val}/{os.path.splitext(os.path.basename(shuffled[gr]))[0]}.png"
        all_data.append(list_t)
    shuffled = all_data
    
    
    part = len(shuffled) // 4
    diff = len(shuffled) - part*4

    part1 = shuffled[0*part:part]
    part2 = shuffled[1*part:2*part]
    part3 = shuffled[2*part:3*part]
    part4 = shuffled[3*part:4*part+diff]
    adobe_data_partition[val] = []

    adobe_data_partition[val].append(part1)
    adobe_data_partition[val].append(part2)
    adobe_data_partition[val].append(part3)
    adobe_data_partition[val].append(part4)

### LINE

pmc_training_set_line = ubpmc_data_partition['line'][0]+ubpmc_data_partition['line'][1]+ubpmc_data_partition['line'][2]
pmc_train_val_set_line = ubpmc_data_partition['line'][0]+ubpmc_data_partition['line'][1]+ubpmc_data_partition['line'][2]
pmc_test_eval_line = ubpmc_data_partition['line'][3]
pmc_train_adobe_test_line = adobe_data_partition['line'][0]\
                     + adobe_data_partition['line'][1]\
                     + adobe_data_partition['line'][2]\
                     + adobe_data_partition['line'][3]
#1.
ubpmc_train_setup_line = {
                           "train": pmc_training_set_line,
                            "train_val":pmc_train_val_set_line,
                            "test_pmc":pmc_test_eval_line,
                            "test_synth":pmc_train_adobe_test_line
}

synth_training_set_line = adobe_data_partition['line'][0]+ adobe_data_partition['line'][1]+adobe_data_partition['line'][2]
synth_train_val_set_line = adobe_data_partition['line'][0]+ adobe_data_partition['line'][1]+adobe_data_partition['line'][2]
synth_test_eval_line = adobe_data_partition['line'][3]

synth_train_ubpmc_test_line = ubpmc_data_partition['line'][0]\
                        +ubpmc_data_partition['line'][1]\
                        +ubpmc_data_partition['line'][2]\
                        +ubpmc_data_partition['line'][3]

#2.
synth_train_setup_line =    {
                            "train":synth_training_set_line,
                            "train_val":synth_train_val_set_line,
                            "test_synth":synth_test_eval_line,
                            "test_pmc":synth_train_ubpmc_test_line
}
#--------------------------

### SCATTER
pmc_training_set_scatter = ubpmc_data_partition['scatter'][0]+ubpmc_data_partition['scatter'][1]+ubpmc_data_partition['scatter'][2]
pmc_train_val_set_scatter = ubpmc_data_partition['scatter'][0]+ubpmc_data_partition['scatter'][1]+ubpmc_data_partition['scatter'][2]
pmc_test_eval_scatter = ubpmc_data_partition['scatter'][3]
pmc_train_adobe_test_scatter = adobe_data_partition['scatter'][0]\
                     + adobe_data_partition['scatter'][1]\
                     + adobe_data_partition['scatter'][2]\
                     + adobe_data_partition['scatter'][3]

#3.
ubpmc_train_setup_scatter = {
                            "train":pmc_training_set_scatter,
                            "train_val":pmc_train_val_set_scatter,
                            "test_pmc":pmc_test_eval_scatter,
                            "test_synth":pmc_train_adobe_test_scatter
}

synth_training_set_scatter = adobe_data_partition['scatter'][0]+ adobe_data_partition['scatter'][1]+adobe_data_partition['scatter'][2]
synth_train_val_set_scatter = adobe_data_partition['scatter'][0]+ adobe_data_partition['scatter'][1]+adobe_data_partition['scatter'][2]
synth_test_eval_scatter = adobe_data_partition['scatter'][3]

synth_train_ubpmc_test_scatter = ubpmc_data_partition['scatter'][0]\
                        +ubpmc_data_partition['scatter'][1]\
                        +ubpmc_data_partition['scatter'][2]\
                        +ubpmc_data_partition['scatter'][3]

#4.
synth_train_setup_scatter = {
                            "train":synth_training_set_scatter,
                            "train_val":synth_train_val_set_scatter,
                            "test_synth":synth_test_eval_scatter,
                            "test_pmc":synth_train_ubpmc_test_scatter
}

#-----------------------

### BOX

pmc_training_set_box = ubpmc_data_partition['vertical_box'][0]+ubpmc_data_partition['vertical_box'][1]\
                      +ubpmc_data_partition['vertical_box'][2]
pmc_train_val_set_box = ubpmc_data_partition['vertical_box'][0]+ubpmc_data_partition['vertical_box'][1]+ubpmc_data_partition['scatter'][2]
pmc_test_eval_box = ubpmc_data_partition['vertical_box'][3]
pmc_train_adobe_test_box = adobe_data_partition['vbox'][0]\
                     + adobe_data_partition['vbox'][1]\
                     + adobe_data_partition['vbox'][2]\
                     + adobe_data_partition['vbox'][3]\
                     + adobe_data_partition['hbox'][0]\
                     + adobe_data_partition['hbox'][1]\
                     + adobe_data_partition['hbox'][2]\
                     + adobe_data_partition['hbox'][3]

#5.
ubpmc_train_setup_box = {
                         "train":pmc_training_set_box,
                         "train_val":pmc_train_val_set_box,
                         "test_pmc":pmc_test_eval_box,
                         "test_synth":pmc_train_adobe_test_box
                         }


synth_training_set_box = adobe_data_partition['hbox'][0]+ adobe_data_partition['hbox'][1]\
                        +adobe_data_partition['hbox'][2]\
                        +adobe_data_partition['vbox'][0]+adobe_data_partition['vbox'][1]\
                        +adobe_data_partition['vbox'][2]

synth_train_val_set_box = adobe_data_partition['hbox'][0]+ adobe_data_partition['hbox'][1]\
                         +adobe_data_partition['hbox'][2]\
                         +adobe_data_partition['vbox'][0]+adobe_data_partition['vbox'][1]\
                         +adobe_data_partition['vbox'][2]

synth_test_eval_box = adobe_data_partition['hbox'][3]+adobe_data_partition['vbox'][3]

synth_train_ubpmc_test_box = ubpmc_data_partition['vertical_box'][0]\
                        +ubpmc_data_partition['vertical_box'][1]\
                        +ubpmc_data_partition['vertical_box'][2]\
                        +ubpmc_data_partition['vertical_box'][3]

#6.
synth_train_setup_box = {
                        "train":synth_training_set_box,
                        "train_val":synth_train_val_set_box,
                        "test_pmc":synth_test_eval_box,
                        "test_synth":synth_train_ubpmc_test_box
                        }


#----------------------------

### BAR 

pmc_training_set_bar = ubpmc_data_partition['vertical_bar'][0]+ubpmc_data_partition['vertical_bar'][1]\
                      +ubpmc_data_partition['vertical_bar'][2]\
                      +ubpmc_data_partition["horizontal_bar"][0]+ubpmc_data_partition["horizontal_bar"][1]\
                      +ubpmc_data_partition["horizontal_bar"][2]

pmc_train_val_set_bar = ubpmc_data_partition['vertical_bar'][0]+ubpmc_data_partition['vertical_bar'][1]\
                      +ubpmc_data_partition['vertical_bar'][2]\
                      +ubpmc_data_partition["horizontal_bar"][0]+ubpmc_data_partition["horizontal_bar"][1]\
                      +ubpmc_data_partition["horizontal_bar"][2]

pmc_test_eval_bar = ubpmc_data_partition['vertical_bar'][3]+ubpmc_data_partition['horizontal_bar'][3]

pmc_train_adobe_test_bar = adobe_data_partition['vStack'][0]\
                     + adobe_data_partition['vStack'][1]\
                     + adobe_data_partition['vStack'][2]\
                     + adobe_data_partition['vStack'][3]\
                     + adobe_data_partition['hStack'][0]\
                     + adobe_data_partition['hStack'][1]\
                     + adobe_data_partition['hStack'][2]\
                     + adobe_data_partition['hStack'][3]\
                     + adobe_data_partition['vGroup'][0]\
                     + adobe_data_partition['vGroup'][1]\
                     + adobe_data_partition['vGroup'][2]\
                     + adobe_data_partition['vGroup'][3]\
                     + adobe_data_partition['hGroup'][0]\
                     + adobe_data_partition['hGroup'][1]\
                     + adobe_data_partition['hGroup'][2]\
                     + adobe_data_partition['hGroup'][3]\

#7.
ubpmc_train_setup_bar = {
                        "train":pmc_training_set_bar,
                        "train_val":pmc_train_val_set_bar,
                        "test_pmc":pmc_test_eval_bar,
                        "test_synth":pmc_train_adobe_test_bar
                        }





synth_training_set_bar = adobe_data_partition['vStack'][0]\
                     + adobe_data_partition['vStack'][1]\
                     + adobe_data_partition['vStack'][2]\
                     + adobe_data_partition['hStack'][0]\
                     + adobe_data_partition['hStack'][1]\
                     + adobe_data_partition['hStack'][2]\
                     + adobe_data_partition['vGroup'][0]\
                     + adobe_data_partition['vGroup'][1]\
                     + adobe_data_partition['vGroup'][2]\
                     + adobe_data_partition['hGroup'][0]\
                     + adobe_data_partition['hGroup'][1]\
                     + adobe_data_partition['hGroup'][2]

synth_train_val_set_bar = adobe_data_partition['vStack'][0]\
                     + adobe_data_partition['vStack'][1]\
                     + adobe_data_partition['vStack'][2]\
                     + adobe_data_partition['hStack'][0]\
                     + adobe_data_partition['hStack'][1]\
                     + adobe_data_partition['hStack'][2]\
                     + adobe_data_partition['vGroup'][0]\
                     + adobe_data_partition['vGroup'][1]\
                     + adobe_data_partition['vGroup'][2]\
                     + adobe_data_partition['hGroup'][0]\
                     + adobe_data_partition['hGroup'][1]\
                     + adobe_data_partition['hGroup'][2]

synth_test_eval_bar = adobe_data_partition['vStack'][3]\
                     + adobe_data_partition['hStack'][3]\
                     + adobe_data_partition['vGroup'][3]\
                     + adobe_data_partition['hGroup'][3]\

synth_train_ubpmc_test_bar = ubpmc_data_partition['vertical_bar'][0]+ubpmc_data_partition['vertical_bar'][1]\
                      +ubpmc_data_partition['vertical_bar'][2] +ubpmc_data_partition['vertical_bar'][3]\
                      +ubpmc_data_partition["horizontal_bar"][0]+ubpmc_data_partition["horizontal_bar"][1]\
                      +ubpmc_data_partition["horizontal_bar"][2]+ubpmc_data_partition['horizontal_bar'][3]

#8.
synth_train_setup_bar = {
                         "train":synth_training_set_bar,
                         "train_val":synth_train_val_set_bar,
                         "test_synth":synth_test_eval_bar,
                         "test_pmc":synth_train_ubpmc_test_bar
                         }



partioned_data = {'ubpmc_train_setup_line':ubpmc_train_setup_line,'synth_train_setup_line':synth_train_setup_line,\
                  'ubpmc_train_setup_scatter':ubpmc_train_setup_scatter,'synth_train_setup_scatter':synth_train_setup_scatter,\
                      'ubpmc_train_setup_box':ubpmc_train_setup_box,'synth_train_setup_box':synth_train_setup_box,\
                          'ubpmc_train_setup_bar':ubpmc_train_setup_bar,'synth_train_setup_bar':synth_train_setup_bar}


# --------------------------------
# with open('scrap/db_split.pickle', 'wb') as handle:
#    pickle.dump(partioned_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('scrap/db_split.pickle', 'rb') as handle:
    b = pickle.load(handle)


print("end")

