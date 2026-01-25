import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


dir = r"champions_clean"

useful_stats = [
    "health", "healthPerLevel", "healthRegen", "healthRegenPerLevel",
    "armor", "armorPerLevel", "magicResistance", "magicResistancePerLevel",
    # stats de ressources ?
    "attackDamage", "attackDamagePerLevel",
    "attackSpeed", "attackSpeedPerLevel",
    "attackRange"
]
roles = ["FIGHTER", "TANK", "ASSASSIN", "MAGE", "SUPPORT", "MARKSMAN"]


rows = []

for file in os.listdir(dir):
    input_path = os.path.join(dir, file)
    row = {}
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    champ_name = file.split(".")[0].lower()
    for keys, values in raw.items():
        if keys == "name":
            row["name"] = values
        elif keys in useful_stats:
            row[keys] = values
        # rajouter les searchTag/roles pour les prédictions
        elif keys == "roles":
            for tag in roles:
                row[tag] = 1 if tag in values else 0

        # récupération des données sur les sorts
                #else:

    rows.append(row)
    

df = pd.DataFrame(rows)
df = df.fillna(0)

df.to_csv("dataframeTest.csv", index=False)


predictions = roles

X = df.drop(columns=predictions + ["name"]).values
y = df[predictions].values
champ_names = df["name"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, tmp_idx = next(msss.split(X, y))
names_train = champ_names[train_idx]
names_tmp = champ_names[tmp_idx]
X_train, X_tmp = X[train_idx], X[tmp_idx]
y_train, y_tmp = y[train_idx], y[tmp_idx]


msss2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(msss2.split(X_tmp, y_tmp))
names_val = names_tmp[val_idx]
names_test = names_tmp[test_idx]
X_val, X_test = X_tmp[val_idx], X_tmp[test_idx]
y_val, y_test = y_tmp[val_idx], y_tmp[test_idx]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)


class GroupedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        return self.activation(self.fc(x))



class Network(nn.Module):

    def __init__(self):
        
        super(Network, self).__init__()

        self.hp_layer = GroupedLayer(4, 3)
        self.tanky_layer = GroupedLayer(4, 2)
        self.attack_layer = GroupedLayer(2, 2)
        self.attackSpeed_layer = GroupedLayer(4, 2)
        self.range_layer = GroupedLayer(1, 1)
        self.msBuff_layer = GroupedLayer(1, 1)
        self.cc_layer = GroupedLayer(1, 1)
        self.ratios_layer = GroupedLayer(2, 3)
        self.mobility_layer = GroupedLayer(2, 3)
        self.allyBuff_layer = GroupedLayer(1, 1)
        
        self.fc1 = nn.Linear(19, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):
        # Découper les features
        hp = x[:, 0:4]
        tanky = x[:, 4:8]
        attack = x[:, 8:10]
        attackSpeed = x[:, [10, 11, 12, 20]]
        range = x[:, 13:14]
        msBuff = x[:, 14:15]
        cc = x[:, 15:16]
        ratios = x[:, [16,18]]
        mobility = x[:, [17,19]]
        allyBuff = x[:, [21]]
        
        # Passer chaque groupe par sa sous-couche
        hp = self.hp_layer(hp)
        tanky = self.tanky_layer(tanky)
        attack = self.attack_layer(attack)
        attackSpeed = self.attackSpeed_layer(attackSpeed)
        range = self.range_layer(range)
        msBuff = self.msBuff_layer(msBuff)
        cc = self.cc_layer(cc)
        ratios = self.ratios_layer(ratios)
        mobility = self.mobility_layer(mobility)
        allyBuff = self.allyBuff_layer(allyBuff)
        
        # Concaténation
        x = torch.cat([hp, tanky, attack, attackSpeed, range, msBuff, cc, ratios, mobility, allyBuff], dim=1)
        
        # MLP final
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Network2(nn.Module): 
    def __init__(self): 
        super(Network2, self).__init__() 
        self.fc1 = nn.Linear(22, 24) 
        self.fc2 = nn.Linear(24, 12) 
        self.fc3 = nn.Linear(12, 6) 
    def forward(self, x): 
        x = self.fc1(x) 
        x = nn.LeakyReLU()(x)
        x = self.fc2(x) 
        x = nn.LeakyReLU()(x) 
        x = self.fc3(x) 
        return x

network = Network()
# Put network to GPU if exists
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.11, weight_decay=1e-4)


def train(model, loss=None, optimizer=None, train_dataloader=None, val_dataloader=None, nb_epochs=20):
    """Training loop"""

    min_val_loss = torch.inf
    train_losses = []
    val_losses = []

    # Iterrate over epochs
    for e in range(nb_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for data, labels in train_dataloader:

            # Transfer data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            # Reset gradients to 0
            optimizer.zero_grad()

            # Forward Pass (on reshaped data)
            #data, labels = reshape_batch([data, labels])
            targets = model(data)

            # Compute training loss
            current_loss = loss(targets, labels)
            train_loss += current_loss.item()

            # Compute gradients
            current_loss.backward()

            # Update weights
            optimizer.step()
        
        # Validation
        val_loss = 0.0

        # Put model in eval mode
        model.eval()

        for data, labels in val_dataloader:

            # Transfer data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass (on reshaped data)
            #data, labels = reshape_batch([data, labels])
            targets = model(data)

            # Compute validation loss
            current_loss = loss(targets, labels)
            val_loss += current_loss.item()
        
        # Prints
        print(f"Epoch {e+1}/{nb_epochs} \
                \t Training Loss: {train_loss/len(train_dataloader):.3f} \
                \t Validation Loss: {val_loss/len(val_dataloader):.3f}")
        
        train_losses.append(train_loss/len(train_dataloader))
        val_losses.append(val_loss/len(val_dataloader))

        # Save model if val loss decreases
        if val_loss < min_val_loss:

            min_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            
    return train_losses, val_losses

train_losses, val_losses = train(model=network, loss=loss, optimizer=optimizer, 
                                 train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                                 nb_epochs=80)


# Plot losses
plt.plot(range(len(train_losses)), train_losses, 
         linewidth=2.0, 
         label='training loss')

plt.plot(range(len(val_losses)), val_losses, 
         linewidth=2.0, 
         label='validation loss')

plt.xlabel("Epochs", size=12)
plt.ylabel("Losses", size=12)
plt.legend()

plt.show()


# --- Prédictions sur test set avec noms ---
network.eval()
with torch.no_grad():
    X_test_device = X_test.to(device)
    logits = network(X_test_device)
    probs = torch.sigmoid(logits)  # Probabilités pour chaque classe
    predictions = (probs > 0.5).int().cpu().numpy()
    y_true = y_test.cpu().numpy()

tags = ["fighter", "mage", "tank", "assassin", "support", "marksman"]

# Affichage avec le nom du champion
for i in range(len(X_test)):
    print(f"Champion : {names_test[i]}")
    for j, tag in enumerate(tags):
        print(f"  {tag:10s} → {probs[i][j].item():.2f}")
    print("Prédit :", [tags[j] for j in range(6) if predictions[i][j] == 1])
    print("Réel  :", [tags[j] for j in range(6) if y_true[i][j] == 1])
    print("-" * 40)
