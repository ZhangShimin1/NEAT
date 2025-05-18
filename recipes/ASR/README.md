# ASR receipt


## 📦 Installation 

Install espnet following the instructions [here](https://github.com/espnet/espnet)

Install espnet in this directory: 
```bash
cd \path\to\this\repo
git clone https://github.com/espnet/espnet

cd <espnet-root>/tools
./setup_miniforge.sh ${CONDA_ROOT} espnet 3.8
conda activate espnet
make
```

kenlm is required for espnet, install it: 
```bash
cd <espnet-root>/tools/installers
./install_kenlm.sh
```
if you have any issues in the installation, please refer to the issue [here](https://github.com/espnet/espnet/issues/6013).

Update the MAIN_ROOT variable in the following files to point to your local ESPnet installation directory:

- `recipes/ASR/egs2/aishell/asr1/path.sh`
- `recipes/ASR/egs2/aishell/asr1/local/path.sh`

For example, if you installed espnet at /home/user/espnet, modify the first line in both files from
MAIN_ROOT=/path/to/espnet to MAIN_ROOT=/home/user/espnet

## 🔨 Usage

Set the HOME_DIR in recipes/ASR/egs2/TEMPLATE/asr1/asr.sh to the absolute path to acouspike source code.

Configure the path of AISHELL dataset in `egs2/aishell/asr1/db.sh`.

Run the following command to train a model:
```bash
cd egs2/aishell/asr1
python run_snn.sh
```

