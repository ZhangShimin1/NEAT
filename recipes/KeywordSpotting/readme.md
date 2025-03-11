# Keyword spotting receipt

## 📦 Training
Run the following command to train the model:
Modify the config file to set the dataset path and other parameters.
```bash
bash run.sh
```

## 📊 Benchmarking Results

### Google Speech Commands v2 (command words only):

Network Architrcture (40-<b>300-300</b>-14)

<table border="0" style="text-align: center;">
    <tr>
        <th>Neuron</th>
        <th>Network</th>
        <th>Parameters (M)</th>
        <th>Accuracy (%)</th>
        <th>Firing Rates</th>
        <th>Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>81.58</td>
        <td>[0.4571, 0.2540]</td>
        <td>38.756</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>86.66</td>
        <td>[0.4732, 0.2538]</td>
        <td>40.059</td>
        <td>63</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.110</td>
        <td><b>94.48</b></td>
        <td>[0.0708, 0.0399]</td>
        <td>12.656</td>
        <td>63</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.828</td>
        <td>93.03</td>
        <td>[0.2412, 0.0153]</td>
        <td>1681.070</td>
        <td>70</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.418</td>
        <td>89.26</td>
        <td>[0.3807, 0.0414]</td>
        <td>1797413</td>
        <td>87</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>84.67</td>
        <td>[0.3115, 0.0625]</td>
        <td>440.698</td>
        <td>69</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.124</td>
        <td>93.89</td>
        <td>[0.3037, 0.3566]</td>
        <td>3336.6</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>88.88</td>
        <td>[0.3462, 0.1082]</td>
        <td>30.861</td>
        <td>65</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.414</td>
        <td>88.93</td>
        <td>[0.5370, 0.2711]</td>
        <td>47.710</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

### Google Speech Commands v2 (all words):

Network Architecture (40-<b>300-300</b>-35)

<table border="0" style="text-align: center;">
    <tr>
        <th>Neuron</th>
        <th>Network</th>
        <th>Parameters (M)</th>
        <th>Accuracy (%)</th>
        <th>Firing Rates</th>
        <th>Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>73.32</td>
        <td>[0.4438, 0.2419]</td>
        <td>37.662</td>
        <td>74</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>80.43</td>
        <td>[0.3902, 0.2345]</td>
        <td>33.310</td>
        <td>76</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.117</td>
        <td><b>91.63</b></td>
        <td>[0.0988, 0.0507]</td>
        <td>14.930</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.834</td>
        <td>89.21</td>
        <td>[0.2738, 0.0193]</td>
        <td>1683.7</td>
        <td>79</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.425</td>
        <td>84.39</td>
        <td>[0.3680, 0.0259]</td>
        <td>1217944</td>
        <td>109</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>72.46</td>
        <td>[0.3719, 0.0247]</td>
        <td>445.54</td>
        <td>72</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.124</td>
        <td>90.82</td>
        <td>[0.3133, 0.3677]</td>
        <td>3337.4</td>
        <td>76</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>83.29</td>
        <td>[0.3380, 0.0718]</td>
        <td>36.177</td>
        <td>74</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.420</td>
        <td>83.77</td>
        <td>[0.5163, 0.2279]</td>
        <td>46.022</td>
        <td>73</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

### Spiking Heidelberg Datasets (SHD):

Network Architecture (700-<b>128-128</b>-20)

<table border="0" style="text-align: center;">
    <tr>
        <th>Neuron</th>
        <th>Network</th>
        <th>Parameters (M)</th>
        <th>Accuracy (%)</th>
        <th>Firing Rates</th>
        <th>Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>75.39</td>
        <td>[0.0971, 0690]</td>
        <td>11.101</td>
        <td>10</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>77.72</td>
        <td>[0.0871, 0.0642]</td>
        <td>10.951</td>
        <td>13</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.110</td>
        <td><b>90.91</b></td>
        <td>[0.0621, 0.0595]</td>
        <td>12.913</td>
        <td>15</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.240</td>
        <td>89.60</td>
        <td>[0.0850, 0.0505]</td>
        <td>314.131</td>
        <td>24</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td>78.24</td>
        <td>[0.0854, 0.0549]</td>
        <td>87858.8</td>
        <td>38</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>77.75</td>
        <td>[0.0882, 0.0673]</td>
        <td>86.335</td>
        <td>14</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.116</td>
        <td>88.48</td>
        <td>[0.3029, 0.3015]</td>
        <td>616.437</td>
        <td>4</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>82.25</td>
        <td>[0.1683, 0.0902]</td>
        <td>12.720</td>
        <td>16</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.240</td>
        <td>87.41</td>
        <td>[0.1055, 0.0853]</td>
        <td>12.372</td>
        <td>15</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

### Spiking Speech Commands (SSC):

Network Architrcture (700-<b>128-128</b>-35)

<table border="0" style="text-align: center;">
    <tr>
        <th>Neuron</th>
        <th>Network</th>
        <th>Parameters (M)</th>
        <th>Accuracy (%)</th>
        <th>Firing Rates</th>
        <th>Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>54.94</td>
        <td>[0.1031, 0.0240]</td>
        <td>11.760</td>
        <td>102</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>57.66</td>
        <td>[0.0869, 0.0240]</td>
        <td>11.521</td>
        <td>124</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.112</td>
        <td>67.86</td>
        <td>[0.1019, 0.0442]</td>
        <td>14.093</td>
        <td>182</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td><b>70.09</b></td>
        <td>[0.0964, 0.0231]</td>
        <td>314.887</td>
        <td>273</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.244</td>
        <td>63.37</td>
        <td>[0.0972, 0.0230]</td>
        <td>48156.8</td>
        <td>428</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>56.84</td>
        <td>[0.0883, 0.0246]</td>
        <td>86.908</td>
        <td>143</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.117</td>
        <td>67.18</td>
        <td>[0.2387, 0.1288]</td>
        <td>616.087</td>
        <td>42</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>60.61</td>
        <td>[0.2837, 0.0462]</td>
        <td>15.008</td>
        <td>197</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td>62.48</td>
        <td>[0.1277, 0.0328]</td>
        <td>13.290</td>
        <td>155</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>