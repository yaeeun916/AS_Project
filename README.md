# AS_Project
TCGA CRC AS Prediction Project from Histologic Tile


This code is a customized version of victoresque’s pytorch-template. Check <https://github.com/victoresque/pytorch-template> for more information.

## 1. Prepare Dataset

* Before running prepare.py, tile dataset zip file must be located at `data_zip_path` (check prepare.py), and TCGA AS label file must be located in `data_info_dir/type` (check config.json)
* Dataset used in this project is JN Kather's Histological images for MSI vs. MSS classification in gastrointestinal cancer, FFPE samples[1], with additional manual removal of non-tumor tiles by pathologist Dr. Younghoon Kim at Catholic University of Korea Seoul St. Mary's Hospital. Only images from colorectal cancer were used.


* Run prepare.py (required only once)
  ```
  python prepare.py -c config.json
  ```
* This will
  * Save tile images under sorted directories as below:
    ```
    data/CRC
    ├── Train/
    │   └── {AS_Label}
    │        └── {tile_image_fname}.png
    └── Test/
        └── {AS_Label}
             └── {tile_image_fname}.png
    ```
  * Create sample_df.csv, tile_df.csv
    ```
    saved/
    ├── models/ 
    ├── log/
    └── data_info/CRC
        └── mmc2.xlsx
        └── sample_df.csv
        └── tile_df.csv
    ```


* When using a different dataset, edit data_zip_path and src_dir of prepare.py and summarizeDF of util.py.


## 2. Train
* Modify the configurations in `.json` config files, then run:


  ```
  python train.py --config config.json
  ```
* This template uses the configurations stored in the json file by default, but by registering custom options you can change some of them using CLI flags. Use the CLI flags for tuning hyper parameters that need to be changed often.
  * `--model` : pretrained model
  * `--fc` : number of fully connected layers to be added on top of the pretrained model 
  * `--low`, `--AS_low` : AS-low threshold
  * `--high`, `--AS_high` : AS-high threshold
  * `--msi_stat` : 'MSS' for MSS tiles, 'MSIMUT' for MSIMUT tiles, 'MSS+MSIMUT' for both MSS and MSIMUT tiles 
  * `--dtype` : cancer type (in this case, 'CRC')


  ```
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
      CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
      CustomArgs(['--model'], type=str, target='arch;type'),
      CustomArgs(['--fc'], type=int, target='arch;args;fc_num'),
      CustomArgs(['--low', '--AS_low'], type=int, target='data_loader;args;AS_low_thresh'),
      CustomArgs(['--high', '--AS_high'], type=int, target='data_loader;args;AS_high_thresh'),
      CustomArgs(['--msi_stat'], type=str, target='data_loader;args;MSIstat'),
      CustomArgs(['--dtype'], type=str, target='data;type')
  ]
  ```
* Using bash shell script programming would be convenient for running combinations of experiments. Modify name of the training session in config files to easily distinguish experiments. 
  

## 3. Test and Evaluate
* You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.
  ```
  python test.py -c {save_dir/models/name/timestamp/config.json} --resume {save_dir/models/name/timestamp/model_best.pth}
  ```
* Use `visualize.ipynb` for visualization of the results.


## References
[1] Kather, Jakob Nikolas. (2019). Histological images for MSI vs. MSS classification in gastrointestinal cancer, FFPE samples [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2530835

