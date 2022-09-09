import torch 
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class CMA: 
    def __init__(self, checkpoint, outcome, num_neuron_batch):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = AutoConfig.from_pretrained(checkpoint)
        self.label2id = {k.lower(): int(v) for k, v in self.config.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device)
        
        self.layers = self.get_layers()
        self.num_layers = self.config.num_hidden_layers * 2 \
                           if self.config.is_encoder_decoder \
                           else self.config.num_hidden_layers
        self.num_neurons = self.config.hidden_size
        
        self.outcome = torch.tensor([self.label2id[outcome]], device=self.device)
        self.num_neuron_batch = num_neuron_batch
        self.neuron_batch_size = None
        self.pos_to_intervene = None
        self.nes_to_intervene = None

    def get_layers(self):
        model_name = self.model.__class__.__name__
        base_model = getattr(self.model, self.model.base_model_prefix)
        if model_name.startswith("Bart"):
            layers = base_model.encoder.layers + base_model.decoder.layers
        elif model_name.startswith("GPT"):
            layers = base_model.h
        elif model_name.startswith("DistilBert"):
            layers = base_model.transformer.layer
        elif model_name.startswith("Albert"):
            layers = []
            for i in range(self.config.num_hidden_layers):
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
                layer_group = base_model.encoder.albert_layer_groups[group_idx]
                layers.append(layer_group)
            layers = torch.nn.ModuleList(layers)
        elif any(model_name.startswith(x) for x in ["Bert", "Roberta", "Deberta", "MobileBert"]):
            layers = base_model.encoder.layer
        return layers
    
    def get_cls_index(self, input_ids):
        model_name = self.model.__class__.__name__
        if model_name.startswith("Bart"):
            position = input_ids.eq(self.config.eos_token_id)[0]
        elif model_name.startswith("GPT2"):
            position = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        elif any(model_name.startswith(x) for x in ["Bert", "Roberta", "Deberta", "DistilBert", "MobileBert"]):
            position = 0
        return position
    
    def get_hidden_states(self, output):
        output_name = output.__class__.__name__
        if output_name.startswith("Seq2Seq"):
            hidden_states = output.encoder_hidden_states[1:] + output.decoder_hidden_states[1:]
        else:
            hidden_states = output.hidden_states[1:]
        return hidden_states
        
    def intervene(self, input, layers, new_hidden_states, neuron_batch=None):
        hooks = []
        def replace_output(module, input, output, new_output, position, layer, neuron_batch):
            is_tuple = True
            if not isinstance(output, tuple):
                is_tuple = False
                output = (output,)
            if self.nes_to_intervene is None:
                for i, n in enumerate(neuron_batch):
                    if position is None:
                         output[0][i, :, n] = new_output[i, :, n]
                    else:
                        output[0][i, position, n] = new_output[i, position, n]
            else:
                for n in self.nes_to_intervene[layer]:
                    if position is None:
                        output[0][0, :, n] = new_output[0, :, n]
                    else:
                        output[0][0, position, n] = new_output[0, position, n]
            if not is_tuple:
                return output[0]
            return output
        
        if neuron_batch is None:
            batch_size = 1
        else:
            batch_size = len(neuron_batch)
        input_batch = {k: v.repeat(batch_size, 1) for k, v in input.items()}
        
        with torch.no_grad():
            for layer in layers:
                new_output = new_hidden_states[layer].repeat(batch_size, 1, 1)
                hooks.append(self.layers[layer].register_forward_hook(
                    lambda m, i, o: replace_output(m, i, o, new_output, self.pos_to_intervene, layer, neuron_batch)))
            probs = torch.nn.functional.softmax(self.model(**input_batch).logits, dim=-1)
            for hook in hooks:
                hook.remove()
        
        return probs
            
    def calculate_intervened_probs(self, input, new_hidden_states):
        if self.nes_to_intervene is None:
            neuron_batch_size = int(self.num_neurons / self.num_neuron_batch)
            neuron_batches = [range(i, i + min(neuron_batch_size, self.num_neurons - i)) 
                          for i in range(0, self.num_neurons, neuron_batch_size)]
            probs = torch.zeros((self.config.num_labels, self.num_layers, self.num_neurons), device=self.device)
            for layer in range(self.num_layers):
                for neuron_batch in neuron_batches:
                    layer_probs = self.intervene(input, [layer], new_hidden_states, neuron_batch)
                    for neuron, ps in zip(neuron_batch, layer_probs):
                        for i in range(self.config.num_labels):
                            probs[i, layer, neuron] = ps[i]
        else:
            probs = torch.zeros((self.config.num_labels, self.num_layers, 1), device=self.device)
            for layer in range(self.num_layers):
                layer_probs = self.intervene(input, [layer], new_hidden_states)
                for i in range(self.config.num_labels):
                    probs[i, layer, 0] = layer_probs[0][i]
        
        return probs 
    
    def calculate_response(self, inputs, intervention):
        if intervention["input"]:
            input = inputs["intervention"]
        else:
            input = inputs["control"]
        
        if intervention["neurons"]:
            output = self.model(**inputs["control"], output_hidden_states=True, output_attentions=True)
            new_hidden_states = self.get_hidden_states(output)
            probs = self.calculate_intervened_probs(input, new_hidden_states)
        else:
            probs = torch.nn.functional.softmax(self.model(**input).logits, dim=-1).flatten()

        outcome_prob = probs.index_select(0, self.outcome).sum(dim=0)    
        return (1 - outcome_prob) / outcome_prob

    def calculate_metric(self, inputs, metric, pos_to_intervene=None, nes_to_intervene=None):
        self.nes_to_intervene = nes_to_intervene
            
        interventions = {}
        if metric == "TE":
            interventions["control"] = {"input": False, "neurons": False}
            interventions["intervention"] = {"input": True, "neurons": False}
        elif metric == "NDE":
            interventions["control"] = {"input": False, "neurons": False}
            interventions["intervention"] = {"input": True, "neurons": True}
        elif metric == "NIE":
            interventions["control"] = {"input": True, "neurons": True}
            interventions["intervention"] = {"input": True, "neurons": False}
        
        odds = {}
        for k, v in interventions.items():
            if pos_to_intervene == "cls":
                self.pos_to_intervene = self.get_cls_index(inputs[k]["input_ids"])
            odds[k] = self.calculate_response(inputs, v)

        return odds["intervention"] / odds["control"]
    
    def encode_data(self, control_data, intervention_data):
        encoded_data = []
        for (c_premise, c_hypothesis), (t_premise, t_hypothesis) in tqdm(zip(control_data, intervention_data), total=len(control_data)):
            encoded_batch = self.tokenizer([c_premise, t_premise], [c_hypothesis, t_hypothesis], padding=True, return_tensors="pt").to(self.device)
            inputs = {
                "control": {k: torch.unsqueeze(v[0], 0) for k, v in encoded_batch.items()},
                "intervention": {k: torch.unsqueeze(v[1], 0) for k, v in encoded_batch.items()}
            }
            encoded_data.append(inputs)
        return encoded_data

    def calculate_TE(self, encoded_data):
        TEs = []
        for inputs in tqdm(encoded_data, total=len(encoded_data)):
            TE = self.calculate_metric(inputs, metric="TE")
            TE = np.log(TE.cpu().detach().numpy())
            TEs.append(TE)

        return TEs

    def calculate_NIE(self, encoded_data):
        NIEs = []
        for inputs in tqdm(encoded_data, total=len(encoded_data)):
            NIE = self.calculate_metric(inputs, metric="NIE")
            NIE = np.log(NIE.cpu().detach().numpy())
            NIEs.append(NIE)
        return NIEs

    def calculate_topk_NIE(self, encoded_data, NIEs, topk_neurons):
        topk_NIEs = []
        for inputs, NIE in tqdm(zip(encoded_data, NIEs), total=len(encoded_data)):
            NIE = torch.from_numpy(NIE)
            indices = torch.topk(NIE, int(self.num_neurons * topk_neurons), dim=1)[1]
            topk_NIE = self.calculate_metric(inputs, metric="NIE", nes_to_intervene=indices)
            topk_NIE = np.log(topk_NIE.cpu().detach().numpy())
            topk_NIEs.append(topk_NIE)

        return topk_NIEs
