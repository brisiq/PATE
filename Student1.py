import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Generator(nn.Module):
    """Simple Fully Connected Generator for MNIST"""
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        
        # We output linearly here because the input MNIST is normalized 
        # around zero with transforms.Normalize((0.1307,), (0.3081,))
        x = self.fc4(x)
        return x.view(-1, 1, 28, 28)

class DiscriminatorStudent(nn.Module):
    """ Discriminator/Student CNN Model: outputs 11 classes (0-9 Digits, 10 Fake)"""
    def __init__(self):
        super(DiscriminatorStudent, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 11) # 11 Classes!

    def forward(self, x, matching=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        features = F.relu(self.fc1(x))
        
        # Feature Matching logic for G's loss
        if matching:
            return features
            
        logits = self.fc2(features)
        return F.log_softmax(logits, dim=1)


class GANStudent:
    """Semi-Supervised GAN Student Training"""
    def __init__(self, args):
        self.args = args
        self.generator = Generator(z_dim=100)
        self.discriminator = DiscriminatorStudent()
        
        # Seperate optimizers for Generator and Discriminator
        self.opt_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def predict(self, data):
        """Use the discriminator to predict real labels (0-9) while ignoring the 11th fake class"""
        self.discriminator.eval()
        with torch.no_grad():
            log_probs = self.discriminator(data)
            # Only consider the first 10 classes (0-9 index) 
            real_probs = log_probs[:, :10]
            return torch.max(real_probs, 1)[1]

    def train_gan(self, labeled_dataset, unlabeled_dataset):
        print("\nStarting GAN Student Training...")
        self.discriminator.train()
        self.generator.train()
        
        for epoch in range(1, self.args.student_epochs + 1):
            
            # Simple circular iterator for the unlabeled dataset
            def unlabeled_generator():
                while True:
                    for d, _ in unlabeled_dataset:
                        yield d

            unlab_iter = unlabeled_generator()
            
            loss_D_total = 0.0
            loss_G_total = 0.0
            
            for (data, target) in labeled_dataset:
                batch_size = data.size(0)
                
                # Fetch a batch of unlabeled data
                data_unlabeled = next(unlab_iter)
                
                # If sizes don't match exactly (e.g. edge cases), skip to prevent tensor mismatches
                if data_unlabeled.size(0) != batch_size:
                    continue
                
                # ===============================
                # 1. Train Discriminator (Student)
                # ===============================
                self.opt_D.zero_grad()
                
                # A) Supervised Loss (from Teachers' noisy labels)
                log_p_lab = self.discriminator(data)
                loss_sup = F.nll_loss(log_p_lab, target)
                
                # B) Unsupervised Real Loss (Predict any real digit 0-9)
                log_p_unl = self.discriminator(data_unlabeled)
                prob_unl = torch.exp(log_p_unl)
                # Sum of probabilities that the image is ANY real class
                prob_real_unl = torch.sum(prob_unl[:, :10], dim=1) 
                loss_unsup_real = -torch.mean(torch.log(prob_real_unl + 1e-6))
                
                # C) Unsupervised Fake Loss (Predict 11th class for Generator data)
                z = torch.randn(batch_size, 100)
                fake_data = self.generator(z).detach() 
                log_p_fake = self.discriminator(fake_data)
                
                fake_targets = torch.ones(batch_size, dtype=torch.long) * 10
                loss_unsup_fake = F.nll_loss(log_p_fake, fake_targets)
                
                # Combine D's objectives
                loss_D = loss_sup + loss_unsup_real + loss_unsup_fake
                loss_D.backward()
                self.opt_D.step()
                loss_D_total += loss_D.item()
                
                # ===============================
                # 2. Train Generator (Counterfeiter)
                # ===============================
                self.opt_G.zero_grad()
                
                z = torch.randn(batch_size, 100)
                fake_data_new = self.generator(z)
                
                # Feature Matching Loss (G tries to make fake features exactly match Real Unlabeled features)
                features_real = self.discriminator(data_unlabeled, matching=True).detach()
                features_fake = self.discriminator(fake_data_new, matching=True)
                
                loss_G = torch.mean(torch.abs(features_real.mean(dim=0) - features_fake.mean(dim=0)))
                loss_G.backward()
                self.opt_G.step()
                loss_G_total += loss_G.item()
                
            print(f"EPOCH: {epoch} | D Loss: {(loss_D_total/len(labeled_dataset)):.4f} | G Loss: {(loss_G_total/len(labeled_dataset)):.4f}")

    def save_model(self):
        torch.save(self.discriminator.state_dict(), "student1_gan_model.pth")
