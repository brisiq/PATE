import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.laplace import Laplace
from util import accuracy
from syft.frameworks.torch.dp import pate


class Teacher:
    """Implementation of teacher models.
       Teacher models are ensemble of models which learns directly disjoint splits of the sensitive data
       The ensemble of teachers are further used to label unlabelled public data on which the student is 
       trained. 
       Args:
           args[Arguments object]: An object of Arguments class with required hyperparameters
           n_teachers[int]: Number of teachers
           epochs[int]: Number of epochs to train each model
    """

    def __init__(self, args, model, n_teachers=1, epsilon=0.5):

        self.n_teachers = n_teachers
        self.model = model
        self.models = {}
        self.args = args
        self.init_models()
        self.noise_eps = 0.1

        # Tracking metrics for Consensus Gap
        self.total_queries = 0
        self.strong_consensus_queries = 0
        self.total_gap_ratio_sum = 0.0

    def init_models(self):
        """Initialize teacher models according to number of required teachers"""

        name = "model_"
        for index in range(0, self.n_teachers):

            model = self.model()
            self.models[name + str(index)] = model

    def addnoise(self, x):
        """Adds Laplacian noise to histogram of counts
           Args:
                counts[torch tensor]: Histogram counts
                epsilon[integer]:Amount of Noise
           Returns:
                counts[torch tensor]: Noisy histogram of counts
        """
        # In DP, the scale factor b of Laplace noise for counting queries is 1.0 / epsilon
        m = Laplace(torch.tensor([0.0]), torch.tensor([1.0 / self.noise_eps]))
        count = x + m.sample()

        return count

    def split(self, dataset):
        """Function to split the dataset into non-overlapping subsets of the data
           Args:
               dataset[torch tensor]: The dataset in the form of (image,label)
           Returns:
               split: Split of dataset
        """

        ratio = int(len(dataset) / self.n_teachers)
        iters = 0
        index = 0
        split = []
        last_batch = ratio * self.n_teachers

        for teacher in range(0, self.n_teachers):

            split.append([])

        for (data, target) in dataset:
            if (iters) % ratio == 0 and iters != 0:

                index += 1

            split[index].append([data, target])
            iters += 1

            if iters == last_batch:
                return split

        return split

    def train(self, dataset):
        """Function to train all teacher models.
           Args:
                dataset[torch tensor]: Dataset used to train teachers in format (image,label)
        """

        split = self.split(dataset)

        for epoch in range(1, self.args.epochs + 1):

            index = 0
            for model_name in self.models:

                print("TRAINING ", model_name)
                print("EPOCH: ", epoch)
                self.loop_body(split[index], model_name, 1)
                index += 1

    def loop_body(self, split, model_name, epoch):
        """Body of the training loop.
           Args:
               split: Split of the dataset which the model has to train.
               model_name: Name of the model.
               epoch: Epoch for which the model is being trained.
        """

        model = self.models[model_name]
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iters = 0
        loss = 0.0
        for (data, target) in split:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            iters += 1
        # Print loss by making using of log intervals
        print("Loss")
        print(loss.item())

    def aggregate(self, model_votes, batch_size):
        """Aggregate model output into a single tensor of votes of all models.
           Args:
                votes: Model output
                n_dataset: Number of datapoints
           Returns:
                counts: Torch tensor with counts across all models    
           """

        counts = torch.zeros([batch_size, 10])
        model_counts = torch.zeros([self.args.n_teachers, batch_size])
        model_index = 0

        for model in model_votes:

            index = 0

            for tensor in model_votes[model]:
                for val in tensor:

                    counts[index][val] += 1
                    model_counts[model_index][index] = val
                    index += 1

            model_index += 1

        return counts, model_counts

    def save_models(self):
        no = 0
        for model in self.models:

            torch.save(self.models[model].state_dict(), "models/" + model)
            no += 1

        print("\n")
        print("MODELS SAVED")
        print("\n")

    def load_models(self):

        path_name = "model_"

        for i in range(0, self.args.n_teachers):

            modelA = self.model()
            self.models[path_name + str(i)] = torch.load("models/" + path_name + str(i))
            self.models[path_name + str(i)] = modelA.load_state_dict()

    def analyze(self, preds, indices, moments=8):

        # Link actual noise_eps and fix delta mathematically (typically 1e-5, not 0.5)
        datadepeps, dataindeps = pate.perform_analysis_torch(
            preds, indices, noise_eps=self.noise_eps, delta=1e-5, moments=moments, beta=0.09
        )
        return datadepeps, dataindeps

    def predict(self, data):
        """Make predictions using Noisy-max using Laplace mechanism.
           Args:
                data: Data for which predictions are to be made
           Returns:
                predictions: Predictions for the data
        """

        model_predictions = {}

        for model in self.models:

            out = []
            output = self.models[model](data)
            output = output.max(dim=1)[1]
            out.append(output)

            model_predictions[model] = out

        counts, model_counts = self.aggregate(model_predictions, len(data))

        # ===== Calculate Prediction Confidence Gap =====
        sorted_counts, _ = torch.sort(counts, dim=1, descending=True)
        gaps = sorted_counts[:, 0] - sorted_counts[:, 1]
        gap_ratios = gaps / float(self.n_teachers)
        
        self.total_queries += len(data)
        self.strong_consensus_queries += (gap_ratios > 0.6).sum().item()
        self.total_gap_ratio_sum += gap_ratios.sum().item()
        # ==============================================

        counts = counts.apply_(self.addnoise)

        predictions = []

        for batch in counts:

            predictions.append(torch.tensor(batch.max(dim=0)[1].long()).clone().detach())

        output = {"predictions": predictions, "counts": counts, "model_counts": model_counts}

        return output

    def print_consensus_metrics(self):
        if self.total_queries == 0:
            print("No queries evaluated yet.")
            return
            
        avg_gap = (self.total_gap_ratio_sum / self.total_queries) * 100
        strong_percent = (self.strong_consensus_queries / self.total_queries) * 100
        
        print("\n" + "="*50)
        print("★ TEACHER ENSEMBLE CONSENSUS REPORT ★")
        print(f"Total Queries Evaluated: {self.total_queries}")
        print(f"Average Gap (Consensus): {avg_gap:.2f}%")
        print(f"Queries with Strong Gap (>60%): {self.strong_consensus_queries} ({strong_percent:.2f}%)")
        print("="*50 + "\n")
