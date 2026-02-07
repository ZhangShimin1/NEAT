# Speaker identification recipe

## Environment activation

```bash
# cd to the root directory
cd NEAT

# activate your virtual environment
source .venv/bin/activate
```

## Data preparation

Download all archive parts form huggingface. Please first install huggingface-cli if you have not done so: https://huggingface.co/docs/huggingface_hub/en/installation.
Please make sure you have logged in to huggingface cli (`hf auth login`) before running the command below.

```bash
# Based on your installation of huggingface-cli, you might need to use `hf` instead of `huggingface-cli`
huggingface-cli download NEAT/Voxceleb1 \
    --repo-type dataset \
    --local-dir </path/to/your/dataset>

hf download NEAT/Voxceleb1 \
    --repo-type=dataset \
    --local-dir </path/to/your/dataset>
```

Concate and unzip the data:

```bash
cat VoxCeleb1_archive.tar.gz.part-* > VoxCeleb1_archive.tar.gz
tar -xzvf VoxCeleb1_archive.tar.gz
```


## 📦 Training
Before starting, ensure the configuration matches your environment:

1. Open `run.py` (or your config file).
2. Update the `dataset_path` to point to your downloaded data (e.g., `/path/to/your/dataset` when downloading).
3. Run the following command to start training:

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

*Note: The energy consumption is measured with the script in neat.utils.energy.py*


