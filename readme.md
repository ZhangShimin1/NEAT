## Auditory porcessing by Spiking Neural Networks

# 2024.12.29 update

**The spiking neuron module offered by Chenxiang Ma has been connected to AcouSpike, following modifications should be noted:**

- The Neuron loops with T internally, thus we need to specify the number of timesteps when instantiating the network. The good new is we dont need to reset the states every epoch during training(TODO: delete the related codes in run.py).
- Each neuron has the same general args, which I have listed in network.py and kws/conf/default.yaml. The specific settings of each neuron are fixed currently.
- TODO: Function that select the spiking neuron type according to the setting in configuration is required.
- TODO: The codes for AAD, SID, SV needs to be updated.
- TODO: Some tiny aspects for dataloaders.
- TODO: PMSN has not been added, the hyper-parameters in each surrogate gradients need default values.
