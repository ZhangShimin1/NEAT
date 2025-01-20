# Installation Instructions

## Step 1: Install ESPnet
Install the ESPnet library using `pip`:
```bash
pip install espnet
```

---

## Step 2: Install SCTK
Install the Speech Recognition Scoring Toolkit (SCTK):

1. Clone the SCTK repository from GitHub:
   ```bash
   git clone https://github.com/usnistgov/SCTK.git
   cd SCTK
   ```

2. Build SCTK:
   ```bash
    make config
    make all
    make check
    make install
    make doc
   ```

3. Navigate to the SCTK binary directory:
   ```bash
   cd /path/to/SCTK/bin
   ```

4. Add the SCTK binary path to your Conda environment:
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'export PATH=/path/to/SCTK/bin:$PATH' > $CONDA_PREFIX/etc/conda/activate.d/sctk.sh
   ```

5. Reload the Conda environment:
   ```bash
   conda deactivate
   conda activate your_environment_name
   ```

---

## Step 3: Modify `asr.sh`

 Update the `HOME_DIR` variable to point to the correct directory:
   ```bash
   HOME_DIR="/path/to/AcouSpike/recipes/espnet_tiny"
   ```

---

## Notes
- Ensure that all required dependencies for ESPnet and SCTK are installed.
- Always verify that the paths specified in the script match your directory structure.

---

For further assistance, refer to the official documentation of [ESPnet](https://espnet.github.io/) and [SCTK](https://github.com/usnistgov/SCTK).
