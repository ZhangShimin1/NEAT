# Auditory attention decoding recipe

## Environment activation

```bash
# cd to the acouspike root directory
cd AcouSpike

# activate your virtual environment
source .venv/bin/activate
```

## 📦 Training
Run the following command to train the model:
Modify the config file to set the dataset path and other parameters.
```bash
bash run.sh
```
Currently, our recipe for auditory attention decoding (2 classes) includes KUL & DTU datasets, supporting the training in the Within-Subject mode. More datasets and Cross-Trial mode will be added in the future.


## 📊 Benchmarking Results
Our benchmark implements a simple two-layer feedforward architecture with the Leaky Integrate-and-Fire (LIF) neuron. To ensure robust evaluation, we assess each subject using 5-fold cross-validation and report the mean accuracy across all folds as the performance metric.

### KULeuven
<table border="0" style="text-align: center;">
    <tr>
        <th>Subject</th>
        <th>Accuracy (%)</th>
        <th>Event Energy (nJ)</th>
        <th>Subject</th>
        <th>Accuracy (%)</th>
        <th>Event Energy (nJ)</th>
    </tr>
    <tr>
        <td><b>1</b></td>
        <td>68.98</td>
        <td>6.092</td>
        <td><b>2</b></td>
        <td>67.22</td>
        <td><6.472/td>
    </tr>
    <tr>
        <td><b>3</b></td>
        <td>71.24</td>
        <td>6.901</td>
        <td><b>4</b></td>
        <td>65.32</td>
        <td>6.573</td>
    </tr>
    <tr>
        <td><b>5</b></td>
        <td>67.34</td>
        <td>6.321</td>
        <td><b>6</b></td>
        <td>60.08</td>
        <td>6.981</td>
    </tr>
    <tr>
        <td><b>7</b></td>
        <td>73.57</td>
        <td>6.221</td>
        <td><b>8</b></td>
        <td>65.28</td>
        <td>6.724</td>
    </tr>
    <tr>
        <td><b>9</b></td>
        <td>69.35</td>
        <td>6.212</td>
        <td><b>10</b></td>
        <td>66.90</td>
        <td>6.395</td>
    </tr>
    <tr>
        <td><b>11</b></td>
        <td>69.35</td>
        <td>6.213</td>
        <td><b>12</b></td>
        <td>73.142</td>
        <td>6.299</td>
    </tr>
    <tr>
        <td><b>13</b></td>
        <td>64.62</td>
        <td>6.705</td>
        <td><b>14</b></td>
        <td>64.84</td>
        <td>6.541</td>
    </tr>
    <tr>
        <td><b>15</b></td>
        <td>68.356</td>
        <td>6.397</td>
        <td><b>16</b></td>
        <td>64.329</td>
        <td>6.824</td>
    </tr>
<table>

### DTU
<table border="0" style="text-align: center;">
    <tr>
        <th>Subject</th>
        <th>Accuracy (%)</th>
        <th>Event Energy (nJ)</th>
        <th>Subject</th>
        <th>Accuracy (%)</th>
        <th>Event Energy (nJ)</th>
    </tr>
    <tr>
        <td><b>1</b></td>
        <td>65.94</td>
        <td>7.851</td>
        <td><b>2</b></td>
        <td>66.09</td>
        <td>7.091</td>
    </tr>
    <tr>
        <td><b>3</b></td>
        <td>71.56</td>
        <td>7.851</td>
        <td><b>4</b></td>
        <td>70.31</td>
        <td>7.991</td>
    </tr>
    <tr>
        <td><b>5</b></td>
        <td>71.46</td>
        <td>7.711</td>
        <td><b>6</b></td>
        <td>82.97</td>
        <td>7.609</td>
    </tr>
    <tr>
        <td><b>7</b></td>
        <td>74.17</td>
        <td>7.581</td>
        <td><b>8</b></td>
        <td>74.90</td>
        <td>7.634</td>
    </tr>
    <tr>
        <td><b>9</b></td>
        <td>71.77</td>
        <td>7.439</td>
        <td><b>10</b></td>
        <td>72.08</td>
        <td>7.971</td>
    </tr>
    <tr>
        <td><b>11</b></td>
        <td>74.18</td>
        <td>7.191</td>
        <td><b>12</b></td>
        <td>69.32</td>
        <td>7.231</td>
    </tr>
    <tr>
        <td><b>13</b></td>
        <td>73.56</td>
        <td>7.985</td>
        <td><b>14</b></td>
        <td>74.20</td>
        <td>7.340</td>
    </tr>
    <tr>
        <td><b>15</b></td>
        <td>76.11</td>
        <td>7.915</td>
        <td><b>16</b></td>
        <td>73.21</td>
        <td>7.836</td>
    </tr>
    <tr>
        <td><b>17</b></td>
        <td>71.43</td>
        <td>7.698</td>
        <td><b>18</b></td>
        <td>70.67</td>
        <td>7.211</td>
    </tr>
<table>