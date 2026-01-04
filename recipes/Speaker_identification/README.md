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

Network Architrcture (40-<b>300-300-300</b>-1251)

<table border="0" style="text-align: center;">
    <tr>
        <th>Neuron</th>
        <th>Network</th>
        <th>Parameters (M)</th>
        <th>Accuracy (%)</th>
        <th>Float Energy (nJ)</th>
        <th>Event Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.569</td>
        <td>50.16</td>
        <td>38.252</td>
        <td>76.563</td>
        <td>122</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.750</td>
        <td>52.87</td>
        <td>60.010</td>
        <td>75.472</td>
        <td>181</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.569</td>
        <td>51.13</td>
        <td>37.286</td>
        <td>75.873</td>
        <td>127</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.594 </td>
        <td>53.01 </td>
        <td>62.147 </td>
        <td>72.214 </td>
        <td>192 </td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.574</td>
        <td>51.73</td>
        <td>49.099</td>
        <td>65.877</td>
        <td>134</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.755</td>
        <td>52.72 </td>
        <td>84.424 </td>
        <td>67.215 </td>
        <td>207 </td>
    </tr>
</table>

*Note: The energy consumption is measured with the script in acouspike.utils.energy.py*


