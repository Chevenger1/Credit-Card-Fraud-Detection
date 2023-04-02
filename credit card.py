import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
data = pd.read_csv('credit_card_data.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Class'], axis=1), data['Class'], test_size=0.2, random_state=42)

# Resample the training set to address class imbalance
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train.values).float()
y_train = torch.tensor(y_train.values).unsqueeze(1).float()
X_test = torch.tensor(X_test.values).float()
y_test = torch.tensor(y_test.values).unsqueeze(1).float()

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the neural network
for epoch in range(100):
    running_loss = 0.0
    
    for i in range(len(X_train)):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(X_train[i])
        loss = criterion(outputs, y_train[i])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    if epoch % 10 == 9:
        print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(X_train)))

# Test the neural network
y_pred = net(X_test)
y_pred = (y_pred > 0.5).float()
accuracy = (y_pred == y_test).float().mean()

print('Accuracy on the test set: %.3f' % accuracy.item())
