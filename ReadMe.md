1. This repository is setup for ChartOCR-Framework simple implementation.
2. To start training use train.py script
    
    2.1 Training can be done on following dataset combinations,
    
         a. --arch

          choices:['ubpmc_line', 'ubpmc_bar','ubpmc_box','ubpmc_scatter',
              'synth_line','synth_bar','synth_box','synth_scatter']

         b. --train_db

          choices: ['synth', 'ubpmc']
         
         c. --test_db

          choices: ['synth', 'ubpmc']

3. To start testing use test.py script 

    3.1 Available flags

        a. --arch

          choices:['ubpmc_line', 'ubpmc_bar','ubpmc_box','ubpmc_scatter',
              'synth_line','synth_bar','synth_box','synth_scatter']
        b. --test_db

          choices: ['synth', 'ubpmc']

4. Checkpoints should be loaded from "ckpt/test" directory

5. Dataset split file is located in "scrap/" directory

6. Output JSONs are stored in "annotation_convert" directory

7. Data Loaders are location in "datasets" dirrectory

8. All dataset files should be made available in "data" directory
    8.1 use "ubpmc" name for UB-PMC dataset
    8.2 use "adobesynth" for Adobe Synth Dataset.

9. Pretrained weights can be found in following url
    https://drive.google.com/drive/folders/1-0cA_PI7iHLllrUL1pcATw7FzcARvVdz?usp=sharing

