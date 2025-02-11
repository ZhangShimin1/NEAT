
class EnergyCalculator:

    def __init__(self, energy_info, eac=0.9, emac=4.6):
        self.n_layers = len(energy_info)
        self.neurons, self.firing_rates, self.dims, self.if_rec = [], [], [], []
        for i in range(self.n_layers):
            self.neurons.append(energy_info[i]['Module Type'])
            self.firing_rates.append(energy_info[i]['Firing Rate'])
            self.dims.append(energy_info[i]['Neuron Number'])
            self.if_rec.append(energy_info[i]['Recurrent'])
        
        self.eac = eac
        self.emac = emac

    def lif_cal(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + dim_out * self.emac  # hard reset
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac  
        
        return energy
    
    def adlif_cal(self, f_in, f_out, dim_in, dim_out, rec):
        energy = 4 * dim_out * self.emac + 2 * dim_out * f_out * self.eac + dim_in * dim_out * f_in * self.eac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac 

        return energy
    
    def ltc_cal(self, f_in, f_out, dim_in, dim_out, rec):
        energy = 5 * dim_in * dim_out * f_in * self.eac + 2 * dim_out * f_out * self.eac + 3 * dim_out * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac
        
        return energy
    
    def readout_cal(self, f_in, dim_in, dim_out):  # SNN + ANN
        energy = dim_in * dim_out * f_in * self.eac + (dim_out + 1) * self.emac  # with bias

        return energy

    def calculate(self):
        energy_consumption = 0

        for syn_index in range(1, self.n_layers):
            neuron, pre_neuron = self.neurons[syn_index], self.neurons[syn_index-1]
            rec = self.if_rec[syn_index]
            f_in, f_out = self.firing_rates[syn_index-1], self.firing_rates[syn_index]
            dim_in, dim_out = self.dims[syn_index-1], self.dims[syn_index]
            if neuron == "lif" or "plif":
                if pre_neuron == neuron or "ann":
                    energy_consumption += self.lif_cal(f_in, f_out, dim_in, dim_out, rec)
                else:  # other spk neurons, indicating the network is heterogenious
                    raise NotImplementedError
            elif neuron == "adlif":
                if pre_neuron == neuron or "ann":
                    energy_consumption += self.adlif_cal(f_in, f_out, dim_in, dim_out, rec)
                else:  # other spk neurons, indicating the network is heterogenious
                    raise NotImplementedError
            elif neuron == "LTC":
                if pre_neuron == neuron or "ann":
                    energy_consumption += self.ltc_cal(f_in, f_out, dim_in, dim_out, rec)
                else:  # other spk neurons, indicating the network is heterogenious
                    raise NotImplementedError
            elif neuron == "ann":  # readout layer
                energy_consumption += self.readout_cal(f_in, dim_in, dim_out)
            else:
                raise NotImplementedError

        return energy_consumption
    

if __name__ == "__main__":
    # ene_info = [{'Module Type': 'LTC', 'Firing Rate': 0.23434, 'Neuron Number': 300, 'Recurrent': False},
    #             {'Module Type': 'LTC', 'Firing Rate': 0.15762, 'Neuron Number': 300, 'Recurrent': False},
    #             {'Module Type': 'LTC', 'Firing Rate': 0.17856, 'Neuron Number': 300, 'Recurrent': False}]
    ene_info = [{'Module Type': 'ann', 'Firing Rate': 1., 'Neuron Number': 128, 'Recurrent': False},
                {'Module Type': 'lif', 'Firing Rate': 0.23434, 'Neuron Number': 256, 'Recurrent': False},
                {'Module Type': 'lif', 'Firing Rate': 0.15762, 'Neuron Number': 64, 'Recurrent': False},
                {'Module Type': 'ann', 'Firing Rate': 1., 'Neuron Number': 20, 'Recurrent': False}]
    ec = EnergyCalculator(energy_info=ene_info)
    print(ec.calculate())