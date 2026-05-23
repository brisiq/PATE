import torch
from Teacher import Teacher
from Model import Model
from data import load_data, NoisyDataset
from util import accuracy, split
from Student1 import GANStudent
import syft as sy
from syft.frameworks.torch.dp import pate


class Arguments:
    def __init__(self):
        self.batchsize = 64
        self.test_batchsize = 10
        self.epochs = 10  # Ensure teachers train long enough to converge on tiny splits
        self.student_epochs = 30
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.n_teachers = 50  # Scaled up for full PATE accuracy
        self.save_model = False


def run_pate_gan():
    args = Arguments()

    print("Loading datasets...")
    train_loader = load_data(True, args.batchsize)
    test_loader = load_data(False, args.batchsize)

    # -----------------------------
    # 1. Train Teacher Models
    # -----------------------------
    print(f"\nTraining {args.n_teachers} Teachers...")
    teacher = Teacher(args, Model, n_teachers=args.n_teachers)
    teacher.train(train_loader)

    # -----------------------------
    # 2. Partition Data for Student
    # -----------------------------
    print("\nPreparing Semi-Supervised Data for GAN Student...")
    # PATE traditionally trains the student ONLY on unseen test/public data
    # Split test dataset into 3 parts:
    # 10% For Teacher Labeling, 70% for Unlabeled, 20% for Validation
    part1, rest = split(test_loader, args.batchsize, split=0.1)
    part2, val = split(rest, args.batchsize, split=0.7)
    
    # Restrict labeled_data_raw to exactly 1000 queries as requested
    labeled_data_raw = []
    queries_collected = 0
    for data, target in part1:
        needed = 1000 - queries_collected
        if needed <= 0:
            break
        labeled_data_raw.append([data[:needed], target[:needed]])
        queries_collected += len(data[:needed])

    unlabeled_data = part2
    
    print(f"\nEvaluating individual accuracy of the {args.n_teachers} Teachers on the query set...")
    teacher_accuracies = []
    for model_name, curr_model in teacher.models.items():
        curr_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in labeled_data_raw:
                pred = curr_model(data).max(1)[1]
                correct += (pred == target).sum().item()
                total += target.size(0)
        teacher_accuracies.append((correct / total) * 100)
    
    print(f"★ Teacher Accuracies - Min: {min(teacher_accuracies):.2f}% | Max: {max(teacher_accuracies):.2f}% | Avg: {(sum(teacher_accuracies)/args.n_teachers):.2f}% ★\n")

    
    # Get Noisy Labels from Teachers for the Small Labeled Set
    n_queries = len(labeled_data_raw) * args.batchsize
    print(f"Student querying teachers for labels on {n_queries} samples...")

    # Intercept teacher queries to compute DP Epsilon bounds
    global_model_counts = []
    global_predictions = []

    def tracker_predict(data):
        out = teacher.predict(data)
        global_model_counts.append(out["model_counts"])
        global_predictions.extend(out["predictions"])
        return out

    labeled_data = NoisyDataset(labeled_data_raw, tracker_predict)

    # -----------------------------
    # 2.5 Print Gap Tracking & Privacy Stats
    # -----------------------------
    teacher.print_consensus_metrics()
    
    flat_targets = torch.cat([target for _, target in labeled_data_raw])
    flat_noisy_preds = torch.stack(global_predictions)
    
    correct_agg = (flat_noisy_preds == flat_targets).sum().item()
    agg_acc = (correct_agg / len(flat_targets)) * 100
    
    print(f"★ Aggregated Noisy Label Accuracy (Teacher Consensus): {agg_acc:.2f}% ★")
    
    print("\nCalculating Differential Privacy bounds (Epsilon)...")
    # model_counts collected are list of (n_teachers, batch_size) -> cat dim=1 -> (n_teachers, n_queries)
    counts_lol = torch.cat(global_model_counts, dim=1)
    # predictions collected are list of (1,) -> stack -> (n_queries,)
    predict_lol = torch.stack(global_predictions)
    
    data_dep_eps, data_ind_eps = teacher.analyze(counts_lol, predict_lol, moments=20)
    print(f"★ Epsilon Bounds: Data-Dependent: {data_dep_eps.item():.4f} | Data-Independent: {data_ind_eps.item():.4f} ★")


    # -----------------------------
    # 3. Train Student (GAN)
    # -----------------------------
    gan_student = GANStudent(args)
    
    # Train Semi-Supervised GAN
    gan_student.train_gan(labeled_data, unlabeled_data)

    # -----------------------------
    # 4. Evaluate GAN Student
    # -----------------------------
    print("\nEvaluating GAN Student Performance on Validation Set...")
    total = 0.0
    correct = 0.0

    for data, target in val:
        predict_lol = gan_student.predict(data)
        correct += float((predict_lol == target).sum().item())
        total += float(target.size(0))

    final_acc = (correct / total) * 100
    print(f"\nPATE-GAN Student Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    run_pate_gan()
