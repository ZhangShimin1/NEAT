import math

class EnergyCalculator:

    def __init__(self, energy_info, eac=0.9, emac=4.6):
        # self.net_dim_in = 75
        # self.n_layers = len(energy_info)
        
        self.firing_rates = energy_info['firing_rates']
        self.module_type = energy_info['neuron_types']
        self.neuron_nums = energy_info['neuron_nums']
        self.recurrent_flag = energy_info['recurrent_flags']
        self.readout_dim = energy_info['readout_dim']
        self.readin_dim = energy_info['readin_dim']
        self.dataset_name = energy_info['dataset_name']
        self.eac = eac
        self.emac = emac

    def lif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + dim_out * (1 + f_out) * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac  
        
        return energy
    
    def adlif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 2 * dim_out * f_out) * self.eac + 5 * dim_out * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac 

        return energy
    
    def tclif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 2 * dim_out * f_out) * self.eac + 2 * dim_out * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac 

        return energy
    
    def celif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + dim_out * f_out) * self.eac + 3 * dim_out * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac 

        return energy
    
    def ltc(self, f_in, f_out, dim_in, dim_out, rec):
        energy = (dim_in * dim_out * f_in + 3 * dim_out * f_out) * self.eac + 4 * dim_out * (1 + dim_in) * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac
        
        return energy
    
    def glif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + \
                 (2 * dim_out**3 + 3 * dim_out**2 + f_out * (2 * dim_out**2 + dim_out**3 + dim_out**4 + dim_out)) * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac

        return energy
    
    def clif(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + (dim_out * f_out + dim_out + dim_out**2) * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac

        return energy

    def pmsn(self, f_in, f_out, dim_in, dim_out, rec):
        energy = dim_in * dim_out * f_in * self.eac + 8 * dim_in * dim_out * self.emac
        if rec:
            energy = energy + dim_in * dim_in * f_in * self.eac

        return energy
    
    def spike_readin(self, spike_input_sparsity, input_dim=700, hid_dim=128):
        """
            For spiking datasets SHD & SSC
        """
        return input_dim * hid_dim * spike_input_sparsity * self.eac
    
    def float_readin(self, dim_in, dim_out):
        """
            For other datasets
        """
        return dim_in * dim_out * self.emac

    def readout(self, f_in, dim_in, dim_out): 
        energy = dim_in * dim_out * f_in * self.eac

        return energy

    def calculate(self):
        event_energy, float_energy = 0, 0
        if self.dataset_name == "shd":
            event_energy += self.spike_readin(spike_input_sparsity=0.1121)
        elif self.dataset_name == "ssc":
            event_energy += self.spike_readin(spike_input_sparsity=0.1195)
        else:
            float_energy += self.float_readin(dim_in=self.readin_dim, dim_out=self.neuron_nums[0])

        event_energy += self.intermediate_snn_layer_energy()

        float_energy += self.readout(f_in=self.firing_rates[-1], dim_in=self.neuron_nums[-1], dim_out=self.readout_dim)

        return event_energy / 1e3, float_energy / 1e3

    def intermediate_snn_layer_energy(self):
        energy_consumption = 0
        dim_in, dim_out = self.neuron_nums
        f_in, f_out = self.firing_rates
        neuron = self.module_type
        rec = self.recurrent_flag

        if neuron.lower() == "rlif":
            energy_consumption += self.lif(f_in, f_out, dim_in, dim_out, rec)
        elif neuron.lower() == "plif":
            energy_consumption += self.lif(f_in, f_out, dim_in, dim_out, rec)
        elif neuron.lower() == "adlif":
            energy_consumption += self.adlif(f_in, f_out, dim_in, dim_out, rec)
        elif neuron.lower() == "ltc":
            energy_consumption += self.ltc(f_in, f_out, dim_in, dim_out, rec)
        elif neuron.lower() == "glif":
            energy_consumption += self.glif(f_in, f_out, dim_in, dim_out, rec)   
        elif neuron.lower() == "tclif":
            energy_consumption += self.tclif(f_in, f_out, dim_in, dim_out, rec)    
        elif neuron.lower() == "clif":
            energy_consumption += self.clif(f_in, f_out, dim_in, dim_out, rec) 
        elif neuron.lower() == "pmsn":
            energy_consumption += self.pmsn(f_in, f_out, dim_in, dim_out, rec) 
        elif neuron.lower() == "celif":
            energy_consumption += self.celif(f_in, f_out, dim_in, dim_out, rec) 
        else:
            raise NotImplementedError

        return energy_consumption
    

if __name__ == "__main__":
    ene_info = {
        'dataset_name': "ssc",
        'neuron_types': "pmsn",
        'firing_rates': [0.2456328272819519, 0.1387682408094406],
        'neuron_nums': [128, 128],
        'recurrent_flags': False,
        'readout_dim': 35,
        'readin_dim': 700
    }
    ec = EnergyCalculator(energy_info=ene_info)
    eve_ene, flo_ene = ec.calculate()
    print(f"Energy consumption: {eve_ene} nJ, {flo_ene} nJ")