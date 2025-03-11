import math

class EnergyCalculator:

    def __init__(self, energy_info, eac=0.9, emac=4.6):
        # self.net_dim_in = 75
        # self.n_layers = len(energy_info)
        
        # for i in range(self.n_layers):
        #     self.neurons.append(energy_info[i]['Module Type'])
        #     self.firing_rates.append(energy_info[i]['Firing Rate'])
        #     self.dims.append(energy_info[i]['Neuron Number'])
        #     self.if_rec.append(energy_info[i]['Recurrent'])

        self.dataset, self.neurons, self.firing_rates, self.dims, self.if_rec = energy_info
        
        self.eac = eac
        self.emac = emac

    def lif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + dim_out * (1 + f_out) * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac  
        
        return energy
    
    def adlif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 2 * dim_out * f_out) * self.eac + 5 * dim_out * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac 

        return energy
    
    def tclif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 2 * dim_out * f_out) * self.eac + 2 * dim_out * self.emac

        return energy
    
    def celif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + dim_out * f_out) * self.eac + 3 * dim_out * self.emac

        return energy
    
    def ltc(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 3 * dim_out * f_out) * self.eac + 4 * dim_out * (1 + dim_in) * self.emac
        
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac
        
        return energy
    
    def glif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + \
                 (2 * dim_out**3 + 3 * dim_out**2 + f_out * (2 * dim_out**2 + dim_out**3 + dim_out**4 + dim_out)) * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac

        return energy
    
    def clif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + (dim_out * f_out + dim_out + dim_out**2) * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac

        return energy

    def pmsn(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + 8 * dim_in * dim_out * self.emac
        if rec:
            energy = energy + dim_out * dim_out * f_out * self.eac

        return energy
    
    def spike_readin(self, spike_input_sparsity, input_dim=700, hid_dim=128):
        """
            For spiking datasets SHD & SSC
        """
        return input_dim * hid_dim * spike_input_sparsity * self.eac

    def readout_cal(self, f_in, dim_in, dim_out): 
        energy = dim_in * dim_out * f_in * self.eac  # without bias

        return energy

    def calculate(self):
        energy_consumption = 0
        dim_in, dim_out = self.dims
        f_data, f_in, f_out = self.firing_rates
        neuron = self.neurons
        rec = self.if_rec

        if neuron == "lif":
            energy_consumption += self.lif(f_in, f_out, dim_in, dim_out, rec)
        if neuron == "plif":
            energy_consumption += self.lif(f_in, f_out, dim_in, dim_out, rec)
        elif neuron == "adlif":
            energy_consumption += self.adlif(f_in, f_out, dim_in, dim_out, rec)
        elif neuron == "ltc":
            energy_consumption += self.ltc(f_in, f_out, dim_in, dim_out, rec)
        elif neuron == "glif":
            energy_consumption += self.glif(f_in, f_out, dim_in, dim_out, rec)   
        elif neuron == "tclif":
            energy_consumption += self.tclif(f_in, f_out, dim_in, dim_out, rec)    
        elif neuron == "clif":
            energy_consumption += self.clif(f_in, f_out, dim_in, dim_out, rec) 
        elif neuron == "pmsn":
            energy_consumption += self.pmsn(f_in, f_out, dim_in, dim_out, rec) 
        elif neuron == "celif":
            energy_consumption += self.celif(f_in, f_out, dim_in, dim_out, rec) 

        if self.dataset in ["shd", "ssc"]:
            energy_consumption += self.spike_readin(f_data)

        # for layer in range(self.n_layers):
        #     neuron = self.neurons[layer]
        #     rec = self.if_rec[layer]
        #     if layer == 0:
        #         f_in = 1
        #         f_out = self.firing_rates[layer]
        #         dim_in = self.net_dim_in
        #         dim_out = self.dims[layer]
        #     else:
        #         f_in, f_out = self.firing_rates[layer-1], self.firing_rates[layer]
        #         dim_in, dim_out = self.dims[layer-1], self.dims[layer]
        #     # print(layer, neuron, rec, f_in, f_out, dim_in, dim_out)

        #     if neuron == "lif":
        #         energy_consumption += self.lif_cal(f_in, f_out, dim_in, dim_out, rec)
        #     if neuron == "plif":
        #         energy_consumption += self.lif_cal(f_in, f_out, dim_in, dim_out, rec)
        #     elif neuron == "adlif":
        #         energy_consumption += self.adlif_cal(f_in, f_out, dim_in, dim_out, rec)
        #     elif neuron == "LTC":
        #         energy_consumption += self.ltc_cal(f_in, f_out, dim_in, dim_out, rec)
        #     elif neuron == "ann":  # readout layer
        #         energy_consumption += self.readout_cal(f_in, dim_in, dim_out)
        #     else:
        #         raise NotImplementedError

        return energy_consumption
    

if __name__ == "__main__":
    ene_info = ["gsc", "lif", [0.1121, 0.4438, 0.2419], [300, 300], False]
    # ene_info = [{'Module Type': 'LTC', 'Firing Rate': 0.31032, 'Neuron Number': 128, 'Recurrent': False},
    #             {'Module Type': 'adlif', 'Firing Rate': 0.23434, 'Neuron Number': 256, 'Recurrent': False},
    #             {'Module Type': 'adlif', 'Firing Rate': 0.15762, 'Neuron Number': 64, 'Recurrent': False},

    #             {'Module Type': 'ann', 'Firing Rate': 1., 'Neuron Number': 20, 'Recurrent': False}]
    ec = EnergyCalculator(energy_info=ene_info)
    energy_pJ = ec.calculate()
    energy_muJ = energy_pJ / 1e3  # converting from picojoules (pJ) to microjoules (µJ)
    print(f"Energy consumption: {energy_muJ} nJ")