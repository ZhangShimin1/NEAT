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

The results on **SHD** are shown in the following table:

| Neuron Type | Config File | Parameters (M) | Accuracy (%) | Energy (nJ) | Time (s/epoch) |
|-------------|-------------|----------------|--------------|-------------|----------------|
| LIF         | lif_shd.yaml | 0.125 | 79.25 |  |  |
| PLIF       | plif_shd.yaml | 0.128 | 80.15 |  | 22.3 |
| adLIF       | adlif_shd.yaml | 0.128 | 88.41 |  |  |
| LTC-LIF       | ltc_shd.yaml | 0.322 | 88.74 |  |  |
| GLIF       | glif_shd.yaml | 0.325 | 82.36 |  | 71.2 |
| CLIF       | clif_shd.yaml | 0.125 | 57.06 |  | 23.45 |
| PMSN       | pmsn_shd.yaml | 0.136 | 85.92 |  | 4.67 |
| TC-LIF       | tclif_shd.yaml | 0.125 | 79.92 |  | 31.2 |
| CELIF       | celif_shd.yaml | 0.323 | 79.49 |  | 27.34 |

The results on **SSC** are shown in the following table:

| Neuron Type | Config File | Parameters (M) | Accuracy (%) | Energy (nJ) | Time (s/epoch) |
|-------------|-------------|----------------|--------------|-------------|----------------|
| LIF         | lif_ssc.yaml | 0.125 | 57.31 |  |  |
| PLIF       | plif_ssc.yaml | 0.128 |  47.23|  |  |
| adLIF       | adlif_ssc.yaml | 0.128 | 59.95 |  |  |
| LTC-LIF       | ltc_ssc.yaml | 0.322 | 72.98 |  |  |
| GLIF       | glif_ssc.yaml | 0.325 |  51.83|  | 584.5 |
| CLIF       | clif_ssc.yaml | 0.125 | 32.26 |  | 198.5 |
| PMSN       | pmsn_ssc.yaml | 0.136 | 62.12 |  | 41.2 |
| TC-LIF       | tclif_ssc.yaml | 0.125 | 43.23 |  | 280.23|
| CELIF       | celif_ssc.yaml | 0.323 | 48.04 |  | 224.7 |

*Note: The energy consumption is measured with the script in acouspike.utils.energy.py*


