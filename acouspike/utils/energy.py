
class EnergyCalculator:

    def __init__(self, energy_info, eac=0.9, emac=4.6):
        self.net_dim_in = 75
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
        if f_in == 1:  # readin layer
            energy = dim_in * dim_out * self.emac + dim_out * self.emac
        else:  # hidden layer
            energy = dim_in * dim_out * f_in * self.eac + dim_out * self.emac  # hard reset

        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac  
        
        return energy
    
    def adlif_cal(self, f_in, f_out, dim_in, dim_out, rec):
        if f_in == 1:  # readin layer
            energy = 4 * dim_out * self.emac + 2 * dim_out * f_out * self.eac + dim_in * dim_out * self.emac
        else:  # hidden layer
            energy = 4 * dim_out * self.emac + 2 * dim_out * f_out * self.eac + dim_in * dim_out * f_in * self.eac
        
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac 

        return energy
    
    def ltc_cal(self, f_in, f_out, dim_in, dim_out, rec):
        if f_in == 1:
            energy = (4 * dim_in * dim_out + 4 * dim_out + dim_in * dim_out) * self.emac + 2 * dim_out * f_out * self.eac
        else:
            energy = (4 * dim_in * dim_out + 4 * dim_out) * self.emac + (dim_in * dim_out * f_in + 2 * dim_out * f_out)* self.eac
        
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac
        
        return energy
    
    def readout_cal(self, f_in, dim_in, dim_out): 
        energy = dim_in * dim_out * f_in * self.eac  # without bias

        return energy

    def calculate(self):
        energy_consumption = 0

        for layer in range(self.n_layers):
            neuron = self.neurons[layer]
            rec = self.if_rec[layer]
            if layer == 0:
                f_in = 1
                f_out = self.firing_rates[layer]
                dim_in = self.net_dim_in
                dim_out = self.dims[layer]
            else:
                f_in, f_out = self.firing_rates[layer-1], self.firing_rates[layer]
                dim_in, dim_out = self.dims[layer-1], self.dims[layer]
            # print(layer, neuron, rec, f_in, f_out, dim_in, dim_out)

            if neuron == "lif":
                energy_consumption += self.lif_cal(f_in, f_out, dim_in, dim_out, rec)
            if neuron == "plif":
                energy_consumption += self.lif_cal(f_in, f_out, dim_in, dim_out, rec)
            elif neuron == "adlif":
                energy_consumption += self.adlif_cal(f_in, f_out, dim_in, dim_out, rec)
            elif neuron == "LTC":
                energy_consumption += self.ltc_cal(f_in, f_out, dim_in, dim_out, rec)
            elif neuron == "ann":  # readout layer
                energy_consumption += self.readout_cal(f_in, dim_in, dim_out)
            else:
                raise NotImplementedError

        return energy_consumption
    

if __name__ == "__main__":
    ene_info = [{'Module Type': 'LTC', 'Firing Rate': 0.23434, 'Neuron Number': 100, 'Recurrent': False},
                {'Module Type': 'LTC', 'Firing Rate': 0.15762, 'Neuron Number': 100, 'Recurrent': False},
                {'Module Type': 'LTC', 'Firing Rate': 0.17856, 'Neuron Number': 300, 'Recurrent': False}]
    # ene_info = [{'Module Type': 'LTC', 'Firing Rate': 0.31032, 'Neuron Number': 128, 'Recurrent': False},
    #             {'Module Type': 'adlif', 'Firing Rate': 0.23434, 'Neuron Number': 256, 'Recurrent': False},
    #             {'Module Type': 'adlif', 'Firing Rate': 0.15762, 'Neuron Number': 64, 'Recurrent': False},
    #             {'Module Type': 'ann', 'Firing Rate': 1., 'Neuron Number': 20, 'Recurrent': False}]
    ec = EnergyCalculator(energy_info=ene_info)
    print(ec.calculate())