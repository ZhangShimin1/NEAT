# Speaker identification receipt


## 📦 Data preparation
run the following command to download the dataset:
```bash
bash download_vox1.sh
``` 
Add the dataset path to the voxceleb1_meta test/train/val.csv

## 📦 Training
Run the following command to train the model:
Modify the config file to set the dataset path and other parameters.
```bash
bash run.sh
```

## 📊 Benchmarking Results

Current results are models trained on the **Voxcelebe1 dev set**. 

The results on **VoxCeleb1 test set** are shown in the following table:

| Neuron Type | Config File | Parameters (M) | Accuracy (%) | Energy (nJ) |
|-------------|-------------|----------------|--------------|-------------|
| LIF         | lif.yaml | 0.569 | 35.8 |  |
| PLIF        | plif.yaml | 0.574 | 34.1 |  |
| adLIF       | adlif.yaml | 0.574 | 24.1 |  |
| LTC-LIF       | ltc.yaml | 1.65 | 14.1 | 28.60 |

*Note: The energy consumption is measured with the script in acouspike.utils.energy.py*


