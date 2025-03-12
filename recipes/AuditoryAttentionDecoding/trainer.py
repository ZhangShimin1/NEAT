import torch
import logging
from tqdm import tqdm
import pandas as pd
from acouspike.utils.monitor import OutputMonitor, cal_firing_rate
from acouspike.utils.energy import EnergyCalculator
from acouspike.models.neuron.base_neuron import BaseNeuron
from acouspike.src.accelerate import gather_object
from acouspike.src.trainer import Trainer as BaseTrainer
logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, *args, cal_FR=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.cal_FR = cal_FR
        # Extract dataset name from args if available
        self.dataset_name = kwargs.get('args', None).data.dataset if hasattr(kwargs.get('args', None), 'data') else None
        if cal_FR:
            self.spike_seq_monitor = OutputMonitor(self.model, BaseNeuron, cal_firing_rate)

    def training_step(self, batch, batch_idx):        
        spect, target = batch
        x = spect.squeeze(1).permute(0, 2, 1).cuda()
        y = target.long().cuda()
        # forward
        logits, states = self.model(x)
        logits = logits.sum(dim=1)
        loss = self.loss_function(logits, y)
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()

        return {
            "loss": loss.detach().cpu().numpy(),
            "accuracy": accuracy.detach().cpu().numpy(),
        }

    def evaluation_step(self, batch, batch_idx, dl_id):
        spect, target = batch
        x = spect.squeeze(1).permute(0, 2, 1).cuda()
        y = target.long().cuda()
        # forward
        logits, states = self.model(x)
        logits = logits.sum(dim=1)
        loss = self.loss_function(logits, y)
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()
        return [{
            "loss": loss.detach().cpu().numpy().item(),
            "test_accuracy": accuracy.detach().cpu().numpy().item(),
        }]
    
    @torch.no_grad()
    def evaluation_loop(self, description: str, gather_step_output: bool = False):
        """Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`."""
        args = self.args

        self.set_models_to_eval_mode()

        if self.is_rank_zero:
            logger.info(f"***** Running {description} *****")
            logger.info(f"Batch size = {args.eval_batch_size}")

        evaluation_output = {}
        if self.cal_FR:
            FR_records = {}
            FR_output = {}
        for dl_idx, (dl_id, dataloader) in enumerate(self.eval_dataloaders.items()):
            dataloader_output = []
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Evaluation on dataloader `{dl_id}`",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=not self.is_rank_zero,
                )
            ):
                """
                It is advised against computing metrics within the `evaluation_epoch_end` method for several reasons:
                    1. Most evaluation metrics are inherently sequential and not parallelizable. Hence, computing them in `evaluation_epoch_end` does not offer a speed advantage during evaluation.
                    2. By not aggregating all outputs for metric calculation at the epoch's end, we reduce the risk of memory overflow, which can occur when gathering results across all processes.
                    3. Calculating the metric score during `evaluation_step` allows for earlier detection of any errors in the code.

                Recommendations for metric calculation:
                    1. Perform immediate metric score calculation within the `evaluation_step` method.
                    2. Accumulate the results at this stage.
                    3. If necessary, compute the average or aggregate metric score in the `evaluation_epoch_end` method.
                """
                self.spike_seq_monitor.clear_recorded_data()
                with torch.no_grad():
                    step_output = self.evaluation_step(batch, batch_idx, dl_id)

                # If `gather_step_output` is True, we will gather the step_output from all processes and return a list of all metric scores.
                if gather_step_output:
                    """
                    Collect the step_output from all processes and return a list of all metric scores. Assume we have two processes:
                    step_output = [
                        {"metric_1": xx, "metric_2": xx, ...},  # process 0
                        {"metric_1": xx, "metric_2": xx, ...},  # process 1
                        {"metric_1": xx, "metric_2": xx, ...},  # process 0
                        {"metric_1": xx, "metric_2": xx, ...},  # process 1
                        ...
                    ]
                    """
                    step_output = gather_object(step_output, dst=0)
                dataloader_output.append(step_output)
            evaluation_output[dl_id] = dataloader_output
            if self.cal_FR:
                for m, i in self.spike_seq_monitor.name_records_index.items():
                    if m not in FR_records.keys():
                        FR_records[m] = {}
                    FR_temp = torch.cat([self.spike_seq_monitor.records[index] for index in self.spike_seq_monitor.name_records_index[m]], dim=0)
                    FR_records[m]['firing_rate'] = FR_temp.mean().detach().cpu().numpy().item()
                    # Get the module instance from the model using the name
                    module = dict(self.model.module.named_modules())[m]
                    FR_records[m]['module_type'] = module.__class__.__name__
                    FR_records[m]['neuron_num'] = module.neuron_num
                    FR_records[m]['recurrent'] = module.recurrent

                # Extract information into separate lists
                extracted_model_info = self.extract_module_data(FR_records)
                energy_calculator = EnergyCalculator(energy_info=extracted_model_info)
                event_energy, float_energy = energy_calculator.calculate()
        

                # Prepare JSON data structure
                model_name = getattr(self.model.module, 'model_name', 'Unknown')
                dataset_name = extracted_model_info['dataset_name'] or 'Unknown'
                
                # Create firing rate data by module
                firing_rate_data = {}
                for module_name, stats in FR_records.items():
                    firing_rate_data[module_name] = {
                        'module_type': stats['module_type'],
                        'firing_rate': stats['firing_rate'],
                        'neuron_num': stats['neuron_num'],
                        'recurrent': stats['recurrent']
                    }
                
                # Create the complete JSON structure
                json_data = {
                    'energy': {
                        'event_energy_nj': event_energy,
                        'float_energy_nj': float_energy,
                        'total_energy_nj': event_energy + float_energy
                    },
                    'firing_rates': firing_rate_data,
                }
                
                # Save as JSON file
                import json
                json_path = self.metrics_dir / f"energy_consumption_epoch_{self.state.epochs_trained}.json"
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
        
    
        return evaluation_output
    
    def extract_module_data(self, FR_records):
        """
        Extract various module information from FR_records into separate lists.
        
        Args:
            FR_records (dict): Dictionary containing information about each module
            
        Returns:
            dict: Dictionary containing extracted lists for each information type and model info
        """
        # Initialize lists for each type of information
        module_names = []
        firing_rates = []
        module_types = []
        neuron_nums = []
        recurrent_flags = []
        
        # Extract information from FR_records
        for module_name, stats in FR_records.items():
            module_names.append(module_name)
            firing_rates.append(stats['firing_rate'])
            module_types.append(stats['module_type'])
            neuron_nums.append(stats['neuron_num'])
            recurrent_flags.append(stats['recurrent'])
        
        # Extract information about the model's dimensions and readout layer
        model_in_dim, model_out_dim = self.get_read_layer_info()
        
        # Extract dataset name from output directory if not stored in the trainer
        dataset_name = self.dataset_name
        if dataset_name is None and hasattr(self, 'metrics_dir'):
            # Try to extract from metrics_dir path which follows the pattern exp/{dataset}/subj-{subject_id}/fold-{fold_id}
            metrics_path_parts = str(self.metrics_dir).split('/')
            if len(metrics_path_parts) >= 2:
                # The dataset name is typically after "exp/" in the path
                for i, part in enumerate(metrics_path_parts):
                    if part == "exp" and i+1 < len(metrics_path_parts):
                        dataset_name = metrics_path_parts[i+1]
                        break
        
        # Return a dictionary with all the extracted lists
        return {
            'firing_rates': firing_rates,
            'neuron_types': module_types[0],
            'neuron_nums': neuron_nums,
            'recurrent_flags': recurrent_flags[0],
            'readin_dim': model_in_dim,
            'readout_dim': model_out_dim,
            'dataset_name': dataset_name
        }
    
    def get_read_layer_info(self):
        """
        Get the input dimension of the first Linear layer and the output dimension of
        the last Linear layer in the model network.
        
        Returns:
            tuple: (input_dim, output_dim) - dimensions of the first and last Linear layers
        """
        # Access the model (unwrap from DDP if necessary)
        model_wrapper = self.model.module if hasattr(self.model, "module") else self.model
        
        # Get all modules
        all_modules = list(model_wrapper.named_modules())
        
        # Find all Linear layers (torch.nn.Linear or custom Linear layers)
        linear_layers = []
        for name, module in all_modules:
            # Check if it's a Linear layer based on class name (more robust than instance checking)
            if "Linear" in module.__class__.__name__:
                linear_layers.append((name, module))
            
            # Also check if it has in_features and out_features attributes (Linear layer interface)
            elif hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                linear_layers.append((name, module))
        
        if not linear_layers:
            logger.warning("No Linear layers found in the model. Returning None for dimensions.")
            return None, None
        
        # Get the first and last Linear layers
        first_linear_name, first_linear = linear_layers[0]
        last_linear_name, last_linear = linear_layers[-1]
        
        logger.info(f"Found first Linear layer: {first_linear_name}, last Linear layer: {last_linear_name}")
        
        # Extract dimensions
        input_dim = getattr(first_linear, 'in_features', None)
        output_dim = getattr(last_linear, 'out_features', None)
        
        # # Log the dimensions
        # logger.info(f"Model read-in dimension (first Linear in_features): {input_dim}")
        # logger.info(f"Model readout dimension (last Linear out_features): {output_dim}")
        
        return input_dim, output_dim
        