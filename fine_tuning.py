import json
import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train
from transformers import PreTrainedTokenizer


# Define a function to load a dataset from Hugging Face
def custom_load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    names: Tuple[str, str, str] = ("train", "valid", "test"),
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        train, valid, test = [
            (
                datasets.create_dataset(dataset[n], tokenizer)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


# Custom Callback to update the plot during training
class PlotUpdateCallback:
    def __init__(self, plot_interval: int = 1):
        self.plot_interval = plot_interval
        self.train_its = []
        self.train_losses = []
        self.validation_its = []
        self.validation_losses = []

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        # Only update the plot based on the interval
        iteration = info["iteration"]
        train_loss = info["train_loss"]

        self.train_its.append(iteration)
        self.train_losses.append(train_loss)

        print(f"Train loss: {train_loss}")
        print(f"Train iteration: {iteration}")

        if iteration % self.plot_interval == 0:
            self.update_plot()

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        # Only update the plot based on the interval
        iteration = info["iteration"]
        val_loss = info["val_loss"]

        self.validation_its.append(iteration)
        self.validation_losses.append(val_loss)
        
        print(f"Validation loss: {val_loss}")
        print(f"Validation iteration: {iteration}")

        if iteration % self.plot_interval == 0:
            self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.train_its, self.train_losses, "-o", label="Train")
        self.ax.plot(
            self.validation_its, self.validation_losses, "-o", label="Validation"
        )
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        plt.draw()  # Redraw the plot
        plt.pause(0.1)  # Pause to allow the plot to update

    def on_train_end(self):
        # After training completes, keep the final plot on screen
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot visible after training is done


prompt = "What is fine-tuning in machine learning?"
model_path = "./Llama-3.2-1B-Instruct"
model, tokenizer = load(model_path)

# interfer with the model
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)


# adapter config
adapter_path = "Llama-3.2-1B-Instruct-adapters"
os.makedirs(adapter_path, exist_ok=True)
adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")

# fine-tuning config
lora_config = {
    "num_layers": 8,
    "lora_parameters": {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
    },
}

# save adapter config
with open(adapter_config_path, "w") as f:
    json.dump(lora_config, f, indent=4)


training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=100,
    steps_per_eval=50,
)

# fine-tune the model
model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Number of trainable parameters: {num_train_params}")
model.train(True)  # set model to train mode


# load dataset
train_set, val_set, test_set = custom_load_hf_dataset(
    data_id="win-wang/Machine_Learning_QA_Collection",
    tokenizer=tokenizer,
    names=("train", "validation", "test"),
)

# train the model
plot_callback = PlotUpdateCallback(plot_interval=1)
# metrics = Metrics()
train(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    optimizer=optim.Adam(learning_rate=1e-5),
    train_dataset=train_set,
    val_dataset=val_set,
    training_callback=plot_callback,
)

# save the model
model_lora, _ = load(model_path, adapter_path=adapter_path)
# now we are adding the adapter path to the model
# Adapters are small trainable layers added to a frozen pre-trained model to fine-tune it efficiently.
response = generate(model_lora, tokenizer, prompt=prompt, verbose=True)
print(response)
