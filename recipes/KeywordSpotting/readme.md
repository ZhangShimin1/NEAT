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
        <th>Float Energy (nJ)</th>
        <th>Event Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>81.58</td>
        <td>56.160</td>
        <td>38.756</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.197</td>
        <td>92.36</td>
        <td>55.948</td>
        <td>66.884</td>
        <td>78</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>86.66</td>
        <td>56.159</td>
        <td>40.059</td>
        <td>63</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.197</td>
        <td>91.62</td>
        <td>55.909</td>
        <td>64.461</td>
        <td>95</td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.110</td>
        <td>94.48</td>
        <td>55.351</td>
        <td>12.656</td>
        <td>63</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.200</td>
        <td>95.93</td>
        <td>55.364</td>
        <td>27.133</td>
        <td>66</td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.828</td>
        <td>93.03</td>
        <td>55.258</td>
        <td>1681.070</td>
        <td>70</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.918</td>
        <td>95.11</td>
        <td>55.282</td>
        <td>1687.046</td>
        <td>172</td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.418</td>
        <td>89.26</td>
        <td>55.356</td>
        <td>1797413</td>
        <td>87</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.509</td>
        <td>93.17</td>
        <td>55.289</td>
        <td>1131061</td>
        <td>103</td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>84.67</td>
        <td>55.436</td>
        <td>440.698</td>
        <td>69</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.197</td>
        <td>85.69</td>
        <td>55.310</td>
        <td>451.297</td>
        <td>72</td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.124</td>
        <td>93.89</td>
        <td>56.548</td>
        <td>3336.603</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.214</td>
        <td>94.06</td>
        <td>56.489</td>
        <td>3363.350</td>
        <td>65</td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.107</td>
        <td>88.88</td>
        <td>55.609</td>
        <td>30.861</td>
        <td>65</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.197</td>
        <td>90.62</td>
        <td>55.521</td>
        <td>58.477</td>
        <td>104</td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.414</td>
        <td>88.93</td>
        <td>56.225</td>
        <td>47.710</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.504</td>
        <td>84.48</td>
        <td>56.414</td>
        <td>90.976</td>
        <td>87</td>
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
        <th>Float Energy (nJ)</th>
        <th>Event Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>73.32</td>
        <td>57.486</td>
        <td>37.662</td>
        <td>74</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.203</td>
        <td>87.62</td>
        <td>56.547</td>
        <td>64.448</td>
        <td>75</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>80.43</td>
        <td>57.416</td>
        <td>33.310</td>
        <td>76</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.203</td>
        <td>88.26</td>
        <td>56.436</td>
        <td>69.073</td>
        <td>75</td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.117</td>
        <td>91.63</td>
        <td>55.679</td>
        <td>14.930</td>
        <td>64</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.207</td>
        <td>93.31</td>
        <td>55.683</td>
        <td>28.893</td>
        <td>67</td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.834</td>
        <td>89.21</td>
        <td>55.382</td>
        <td>1683.745</td>
        <td>79</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.925</td>
        <td>91.96</td>
        <td>55.446</td>
        <td>1689.889</td>
        <td>86</td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.425</td>
        <td>84.39</td>
        <td>55.444</td>
        <td>1217944.069</td>
        <td>109</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.515</td>
        <td>88.85</td>
        <td>55.385</td>
        <td>980280.333</td>
        <td>104</td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>72.46</td>
        <td>55.433</td>
        <td>445.54</td>
        <td>72</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.203</td>
        <td>89.01</td>
        <td>55.430</td>
        <td>477.789</td>
        <td>76</td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.124</td>
        <td>90.82</td>
        <td>58.675</td>
        <td>3337.412</td>
        <td>76</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.220</td>
        <td>91.11</td>
        <td>58.400</td>
        <td>3363.341</td>
        <td>79</td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.113</td>
        <td>83.29</td>
        <td>55.879</td>
        <td>36.177</td>
        <td>66</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.203</td>
        <td>86.80</td>
        <td>55.937</td>
        <td>51.878</td>
        <td>69</td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.420</td>
        <td>83.77</td>
        <td>57.354</td>
        <td>46.022</td>
        <td>73</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.511</td>
        <td>78.01</td>
        <td>56.233</td>
        <td>81.058</td>
        <td>73</td>
    </tr>
</table>

SCNN (Spiking ResNet18 with LIF neuron, Parameters: 11.2M)

<table border="0" style="text-align: center;">
    <tr>
        <th>Time Steps</th>
        <th>Accuracy (%)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td>2</td>
        <td>96.11</td>
        <td>167</td>
    </tr>
    <tr>
        <td>4</td>
        <td>94.79</td>
        <td>182</td>
    </tr>
    <tr>
        <td>6</td>
        <td>96.01</td>
        <td>191</td>
    </tr>
<table>

### Spiking Heidelberg Datasets (SHD):

Network Architecture (700-<b>128-128</b>-20)

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
        <td>0.109</td>
        <td>75.39</td>
        <td>0.159</td>
        <td>11.101</td>
        <td>10</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.125</td>
        <td>86.34</td>
        <td>0.180</td>
        <td>13.849</td>
        <td>15</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>77.72</td>
        <td>0.148</td>
        <td>10.951</td>
        <td>13</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.125</td>
        <td>82.45</td>
        <td>0.132</td>
        <td>13.791</td>
        <td>17</td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.110</td>
        <td>90.91</td>
        <td>0.137</td>
        <td>12.913</td>
        <td>15</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.127</td>
        <td>90.97</td>
        <td>0.128</td>
        <td>14.072</td>
        <td>18</td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.240</td>
        <td>89.60</td>
        <td>0.116</td>
        <td>314.131</td>
        <td>24</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.257</td>
        <td>89.70</td>
        <td>0.123</td>
        <td>316.314</td>
        <td>30</td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td>78.24</td>
        <td>0.126</td>
        <td>87858.801</td>
        <td>38</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.258</td>
        <td>83.87</td>
        <td>0.145</td>
        <td>98369.607</td>
        <td>38</td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>77.75</td>
        <td>0.155</td>
        <td>86.335</td>
        <td>14</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.125</td>
        <td>79.92</td>
        <td>0.139</td>
        <td>88.585</td>
        <td>18</td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.116</td>
        <td>88.48</td>
        <td>0.695</td>
        <td>616.437</td>
        <td>4</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.133</td>
        <td>87.80</td>
        <td>0.703</td>
        <td>612.240</td>
        <td>6</td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.109</td>
        <td>82.25</td>
        <td>0.208</td>
        <td>12.720</td>
        <td>16</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.125</td>
        <td>83.52</td>
        <td>0.251</td>
        <td>18.742</td>
        <td>23</td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.240</td>
        <td>87.41</td>
        <td>0.197</td>
        <td>12.372</td>
        <td>15</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.256</td>
        <td>88.66</td>
        <td>0.178</td>
        <td>14.603</td>
        <td>21</td>
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
        <th>Float Energy (nJ)</th>
        <th>Event Energy (nJ)</th>
        <th>Time (s/epoch)</th>
    </tr>
    <tr>
        <td rowspan="2">LIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>54.94</td>
        <td>0.097</td>
        <td>11.760</td>
        <td>102</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.127</td>
        <td>66.12</td>
        <td>0.010</td>
        <td>16.533</td>
        <td>122</td>
    </tr>
    <tr>
        <td rowspan="2">PLIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>57.66</td>
        <td>0.097</td>
        <td>11.521</td>
        <td>124</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.127</td>
        <td>66.87</td>
        <td>0.097</td>
        <td>17.670</td>
        <td>141</td>
    </tr>
    <tr>
        <td rowspan="2">adLIF</td>
        <td>Feedforward</td>
        <td>0.112</td>
        <td>67.86</td>
        <td>0.178</td>
        <td>14.093</td>
        <td>162</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.129</td>
        <td>70.66</td>
        <td>0.149</td>
        <td>14.455</td>
        <td>168</td>
    </tr>
    <tr>
        <td rowspan="2">LTC-LIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td>70.09</td>
        <td>0.093</td>
        <td>314.887</td>
        <td>273</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.259</td>
        <td>71.74</td>
        <td>0.090</td>
        <td>317.645</td>
        <td>231</td>
    </tr>
    <tr>
        <td rowspan="2">GLIF</td>
        <td>Feedforward</td>
        <td>0.244</td>
        <td>63.37</td>
        <td>0.093</td>
        <td>48156.823</td>
        <td>428</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.260</td>
        <td>68.86</td>
        <td>0.096</td>
        <td>49433.453</td>
        <td>364</td>
    </tr>
    <tr>
        <td rowspan="2">CLIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>56.84</td>
        <td>0.099</td>
        <td>86.908</td>
        <td>143</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.127</td>
        <td>69.13</td>
        <td>0.085</td>
        <td>90.247</td>
        <td>146</td>
    </tr>
    <tr>
        <td rowspan="2">PMSN</td>
        <td>Feedforward</td>
        <td>0.117</td>
        <td>67.18</td>
        <td>0.519</td>
        <td>616.087</td>
        <td>28</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.134</td>
        <td>66.81</td>
        <td>0.560</td>
        <td>616.190</td>
        <td>32</td>
    </tr>
    <tr>
        <td rowspan="2">TC-LIF</td>
        <td>Feedforward</td>
        <td>0.111</td>
        <td>60.61</td>
        <td>0.186</td>
        <td>15.008</td>
        <td>197</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.127</td>
        <td>63.76</td>
        <td>0.189</td>
        <td>19.591</td>
        <td>174</td>
    </tr>
    <tr>
        <td rowspan="2" class="celif">CELIF</td>
        <td>Feedforward</td>
        <td>0.242</td>
        <td>62.48</td>
        <td>0.132</td>
        <td>13.290</td>
        <td>155</td>
    </tr>
    <tr>
        <td>Recurrent</td>
        <td>0.258</td>
        <td>64.81</td>
        <td>0.112</td>
        <td>16.406</td>
        <td>167</td>
    </tr>
</table>