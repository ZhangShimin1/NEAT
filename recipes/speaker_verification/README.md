# Speaker verification recipe

## Environment activation

```bash
# cd to the acouspike root directory
cd AcouSpike

# activate your virtual environment
source .venv/bin/activate
```

## Data preparation

Download all archive parts form huggingface. Please first install huggingface-cli if you have not done so: https://huggingface.co/docs/huggingface_hub/en/installation.
Please make sure you have logged in to huggingface cli (`hf auth login`) before running the command below.

```bash
# Based on your installation of huggingface-cli, you might need to use `hf` instead of `huggingface-cli`
huggingface-cli download Acouspike/Voxceleb1 \
    --repo-type dataset \
    --local-dir </path/to/your/dataset>

hf download Acouspike/Voxceleb1 \
    --repo-type=dataset \
    --local-dir </path/to/your/dataset>
```

Concate and unzip the data:

```bash
cat VoxCeleb1_archive.tar.gz.part-* > VoxCeleb1_archive.tar.gz
tar -xzvf VoxCeleb1_archive.tar.gz
```

Build the dataset metadata from the dataset. you need to change `VOX_ROOT` to the dataset path you just unzipped.

```bash
bash prep_data.sh
```

## 📦 Training
Run the following command to train the model. You need to modify `run.sh` to change:

- `gpu_ids`: specify which GPU(s) to use.
- Other hyperparameters in `conf/*.yaml` if needed.

```bash
bash run.sh
```

## 📊 Benchmarking Results

Current results are models trained on the **Voxcelebe1-O set**. 

Network Architrcture (40-<b>300-300-300-300</b>-256)



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
        <td>0.372</td>
        <td>14.514</td>
        <td>34.337</td>
        <td>68.011</td>
        <td>701</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.643</td>
        <td>14.610 </td>
        <td>60.40 </td>
        <td>62.85 </td>
        <td>753</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.372</td>
        <td>15.493 </td>
        <td>27.89 </td>
        <td>58.74 </td>
        <td>719</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.643</td>
        <td>13.7717 </td>
        <td>49.71 </td>
        <td>62.71 </td>
        <td>819</td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.379</td>
        <td>13.642</td>
        <td>57.969</td>
        <td>61.704</td>
        <td>907</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.650</td>
        <td>13.30 </td>
        <td>91.04 </td>
        <td>61.04 </td>
        <td>948 </td>
    </tr>
</table>

*Note: The energy consumption is measured with the script in acouspike.utils.energy.py*


