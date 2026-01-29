from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import  multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

cpt = 0
cpt1 = 0
spells_without_datavalues = []

dir = r"out_clean"

useful_stats = [
    "baseHP", "hpPerLevel", "baseStaticHPRegen", "hpRegenPerLevel",
    "baseArmor", "armorPerLevel", "baseSpellBlock", "spellBlockPerLevel",
    # stats de ressources ?
    "baseDamage", "damagePerLevel",
    "attackSpeed", "attackSpeedRatio", "attackSpeedPerLevel",
    "attackRange"
]

'''
rows = []

for file in os.listdir(dir):
    input_path = os.path.join(dir, file)
    row = {}
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    champ_name = file.split(".")[0].lower()
    for spell, data in raw.items():
        row["name"] = champ_name
        if "Root" in spell:
            cpt1 += len(raw)-1
            for stat in useful_stats:
                row[stat] = data.get(stat, 0)
            # rajouter les searchTag/roles pour les prédictions
            for tag in ["fighter", "mage", "tank", "assassin", "support", "marksman"]:
                row[tag] = 1 if tag in data.get("searchTags", "0") or tag in data.get("searchTagsSecondary", "0") else 0
        else:
            has_datavalues = False
            for key, value in data.items():
                if key == "mSpellTags":
                    for elem in value:
                        if "CC" in elem:
                            tmp = row.get("CC", 0)
                            row["CC"] = tmp + 1
                        if "Boon" in elem:
                            tmp = row.get("AllyBuff", 0)
                            row["AllyBuff"] = tmp + 1
                        if "MoveBlock" in elem or "Teleport" in elem:
                            tmp = row.get("Dash/Blink", 0)
                            row["Dash/Blink"] = tmp + 1
                elif key == "DataValues":
                    has_datavalues = True
                    cpt += 1
                    for elem in value:
                        for name, val in elem.items():
                            if "ADRatio" in val:
                                tmp = row.get("ADRatio", 0)
                                row["ADRatio"] = tmp + 1
                            elif "APRatio" in val:
                                tmp = row.get("APRatio", 0)
                                row["APRatio"] = tmp + 1
                            elif "ArmorRatio" in val or "MRRatio" in val or "BonusHealthRatio" in val:
                                tmp = row.get("TankyRatio", 0)
                                row["TankyRatio"] = tmp + 1
                            elif "MovementSpeed" in val and val != "MovementSpeedReduction":
                                tmp = row.get("MovementSpeedBuff", 0)
                                row["MovementSpeedBuff"] = tmp + 1
                            elif "StealthDuration" in val:
                                tmp = row.get("Stealth", 0)
                                row["Stealth"] = tmp + 1
                            elif "AttackSpeed" in val or "ASBuff" in val:
                                tmp = row.get("AttackSpeedBuff", 0)
                                row["AttackSpeedBuff"] = tmp + 1
            if not has_datavalues:
                spells_without_datavalues.append(f"{champ_name} - {spell}")
                #elif key == "mSpellCalculations":
                #elif key == "cooldownTime":

                #else:

    rows.append(row)
    
print("nombre de datavalues : " + str(cpt))
print("nombre de spells " + str(cpt1))
print("Nombre de spells sans DataValues :", len(spells_without_datavalues))
for s in spells_without_datavalues:
    print(s)

df = pd.DataFrame(rows)
df = df.fillna(0)

df.to_csv("dataframe.csv", index=False)'''

tags = ["fighter", "mage", "tank", "assassin", "support", "marksman"]

cols_to_scale = [
    "baseHP", "hpPerLevel", "baseStaticHPRegen", "hpRegenPerLevel",
    "baseArmor", "armorPerLevel", "baseSpellBlock", "spellBlockPerLevel",
    "baseDamage", "damagePerLevel",
    "attackSpeed", "attackSpeedRatio", "attackSpeedPerLevel",
    "attackRange"
]

# Colonnes à NE PAS normaliser
cols_no_scale = [
    "MovementSpeedBuff", "CC", "Dash/Blink", "AttackSpeedBuff", "AllyBuff", "Stealth", "selfProtection"
]

df = pd.read_csv("dataframeMain.csv")

# charger les données
X = df.drop(columns=tags + ["name", "ADRatio", "APRatio", "TankyRatio"])
#X = df.drop(columns=tags + ["name", "TankyRatio"])
y = df[tags].values
champ_names = df["name"].values

scaler = StandardScaler()
#X = scaler.fit_transform(X)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), cols_to_scale),
        ("cat", "passthrough", cols_no_scale)
    ]
)

X = preprocessor.fit_transform(X)

# découpage des données
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.40,
    random_state=11
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
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.dropout(self.activation(self.fc(x)))


class Network(nn.Module):

    def __init__(self):
        
        super(Network, self).__init__()

        self.hp_layer = GroupedLayer(4, 3)
        self.tanky_layer = GroupedLayer(4, 3)
        self.attack_layer = GroupedLayer(2, 3)
        self.attackSpeed_layer = GroupedLayer(4, 3)
        self.range_layer = GroupedLayer(1, 1)
        self.msBuff_layer = GroupedLayer(1, 2)
        self.cc_layer = GroupedLayer(1, 2)
        #self.ratios_layer = GroupedLayer(2, 3)
        self.mobility_layer = GroupedLayer(2, 3)
        self.allyBuff_layer = GroupedLayer(1, 2)
        
        self.fc1 = nn.Linear(22, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 6)

    def forward(self, x):
        # Découper les features
        hp = x[:, 0:4]
        tanky = x[:, 4:8]
        attack = x[:, 8:10]
        attackSpeed = x[:, [10, 11, 12, 17]]
        range = x[:, 13:14]
        msBuff = x[:, 14:15]
        cc = x[:, 15:16]
        #ratios = x[:, [16,18]]
        mobility = x[:, [16,19]]
        allyBuff = x[:, [18]]
        
        # Passer chaque groupe par sa sous-couche
        hp = self.hp_layer(hp)
        tanky = self.tanky_layer(tanky)
        attack = self.attack_layer(attack)
        attackSpeed = self.attackSpeed_layer(attackSpeed)
        range = self.range_layer(range)
        msBuff = self.msBuff_layer(msBuff)
        cc = self.cc_layer(cc)
        #ratios = self.ratios_layer(ratios)
        mobility = self.mobility_layer(mobility)
        allyBuff = self.allyBuff_layer(allyBuff)
        
        # Concaténation
        #x = torch.cat([hp, tanky, attack, attackSpeed, range, msBuff, cc, ratios, mobility, allyBuff], dim=1)
        x = torch.cat([hp, tanky, attack, attackSpeed, range, msBuff, cc, mobility, allyBuff], dim=1)
        
        # MLP final
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Network1(nn.Module):

    def __init__(self):
        
        super(Network1, self).__init__()

        self.hp_layer = GroupedLayer(4, 5)
        self.tanky_layer = GroupedLayer(5, 5)
        self.attack_layer = GroupedLayer(2, 4)
        self.attackSpeed_layer = GroupedLayer(4, 5)
        self.range_layer = GroupedLayer(1, 2)

        self.msBuff_layer = GroupedLayer(1, 2)
        self.cc_layer = GroupedLayer(1, 2)
        self.mobility_layer = GroupedLayer(2, 4)
        self.allyBuff_layer = GroupedLayer(1, 2)

        self.stats_layer = GroupedLayer(21, 28)
        self.abilities_layer = GroupedLayer(10, 14)
        
        self.fc1 = nn.Linear(42, 50)
        self.fc2 = nn.Linear(50, 24)
        self.fc3 = nn.Linear(24, 6)

    def forward(self, x):
        # Découper les features
        hp = x[:, 0:4]
        tanky = x[:, [4, 5, 6, 7, 20]]
        attack = x[:, 8:10]
        attackSpeed = x[:, [10, 11, 12, 17]]
        range = x[:, 13:14]
        msBuff = x[:, 14:15]
        cc = x[:, 15:16]
        mobility = x[:, [16,19]]
        allyBuff = x[:, [18]]
        
        # Passer chaque groupe par sa sous-couche
        hp = self.hp_layer(hp)
        tanky = self.tanky_layer(tanky)
        attack = self.attack_layer(attack)
        attackSpeed = self.attackSpeed_layer(attackSpeed)
        range = self.range_layer(range)
        msBuff = self.msBuff_layer(msBuff)
        cc = self.cc_layer(cc)
        mobility = self.mobility_layer(mobility)
        allyBuff = self.allyBuff_layer(allyBuff)
        
        # Concaténation
        stats = torch.cat([hp, tanky, attack, attackSpeed, range], dim=1)
        abilities = torch.cat([ msBuff, cc, mobility, allyBuff], dim=1)

        stats = self.stats_layer(stats)
        abilities = self.abilities_layer(abilities)

        x = torch.cat([stats, abilities], dim=1)
        
        # MLP final
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x


class Network2(nn.Module): 
    def __init__(self): 
        super(Network2, self).__init__() 
        self.fc1 = nn.Linear(21, 32) 
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 6) 
    def forward(self, x): 
        x = self.fc1(x) 
        x = nn.LeakyReLU()(x)
        x = self.fc2(x) 
        x = nn.LeakyReLU()(x) 
        x = self.fc3(x) 
        return x
    

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
        if (e+1) % 10 == 0:
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

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runXTest(x, nb_epochs=80, base_seed=42):
    all_train_losses = []
    all_val_losses = []

    for i in range(x):
        print(f"\n===== RUN {i+1}/{x} =====")

        # 🔁 Seed différent mais contrôlé
        set_seed(base_seed + i)

        network = Network1()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )

        train_losses, val_losses = train(
            model=network,
            loss=loss_fn,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            nb_epochs=nb_epochs
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    # → shape (x, nb_epochs)
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)

    # Moyennes & std
    train_mean = all_train_losses.mean(axis=0)
    train_std = all_train_losses.std(axis=0)

    val_mean = all_val_losses.mean(axis=0)
    val_std = all_val_losses.std(axis=0)

    # Minima globaux
    train_min = train_mean.min()
    train_min_epoch = train_mean.argmin()

    val_min = val_mean.min()
    val_min_epoch = val_mean.argmin()

    print("\n===== RÉSULTATS MOYENS =====")
    print(f"Train min loss : {train_min:.4f} à l'epoch {train_min_epoch}")
    print(f"Val   min loss : {val_min:.4f} à l'epoch {val_min_epoch}")

    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "val_mean": val_mean,
        "val_std": val_std,
        "train_min": train_min,
        "train_min_epoch": train_min_epoch,
        "val_min": val_min,
        "val_min_epoch": val_min_epoch,
    }

def plot(x=5, nb_epochs=80):
    results = runXTest(x=5, nb_epochs=80)
    epochs = range(len(results["train_mean"]))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, results["train_mean"], label="Train (mean)")
    plt.fill_between(
        epochs,
        results["train_mean"] - results["train_std"],
        results["train_mean"] + results["train_std"],
        alpha=0.2
    )

    plt.plot(epochs, results["val_mean"], label="Val (mean)")
    plt.fill_between(
        epochs,
        results["val_mean"] - results["val_std"],
        results["val_mean"] + results["val_std"],
        alpha=0.2
    )

    plt.axvline(results["val_min_epoch"], linestyle="--", alpha=0.5, label="Val min epoch")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Moyenne des losses sur plusieurs runs")
    plt.show()

def predict(network=Network1(), loss=nn.BCEWithLogitsLoss()):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    optimizer = torch.optim.Adam(network.parameters(),lr=0.001,weight_decay=1e-4)

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

    network.eval()
    
    with torch.no_grad():
        X_test_device = X_test.to(device)
        logits = network(X_test_device)
        probs = torch.sigmoid(logits)  # Probabilités pour chaque classe
        

    champion_predictions = dict()

    for i in range(len(X_test)):
        champion_predictions[names_test[i]] = [probs[i][j].item() for j in range(len(tags))]

    return champion_predictions

def showPrediction(champion_prediction : dict):

    # Affichage avec le nom du champion
    for i in range(len(X_test)):

        champ_name = names_test[i]
        print(f"Champion : {champ_name}")

        for j, tag in enumerate(tags):
            print(f"  {tag:10s} → {champion_prediction[champ_name][j]:0.2f}")

        predictions = [(champion_prediction[champ_name][i] > 0.5) for i in range (6)]
        print("Prédit :", [tags[k] for k in range(6) if predictions[k]])

        y_true = y_test.cpu().numpy()
        print("Réel  :", [tags[j] for j in range(6) if y_true[i][j] == 1])
        print("-" * 40)
        
def chart(champion_pred, champion_name):
    
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    champion_pred += champion_pred[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Grille hexagonale
    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    for level in grid_levels:
        vertices = [[angle, level] for angle in angles[:-1]]
        vertices.append(vertices[0])
        vertices = np.array(vertices)
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=0.5, alpha=0.3)
    
    # Axes radiaux
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], 'k-', linewidth=0.5, alpha=0.3)

    ax.plot(angles, champion_pred, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, champion_pred, alpha=0.25, color='#1f77b4')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tags, size=12)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)

    ax.spines['polar'].set_visible(False)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)  # Sens horaire

    plt.title(champion_name, size=16, y=1.08, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.show()

def analyze_predictions(predictions_dict, threshold=0.5):
    
    champion_names = list(predictions_dict.keys())
    y_pred_probs = np.array([predictions_dict[name] for name in champion_names])
    y_true = np.array(y_test)
    
    # Binarise les prédictions avec le seuil
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Matrices de confusion pour chaque label
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    _ , axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, (tag, cm) in enumerate(zip(tags, mcm)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Non', 'Oui'], yticklabels=['Non', 'Oui'],
                    cbar_kws={'label': 'Nombre'})
        axes[idx].set_title(f'{tag}', fontweight='bold', fontsize=14)
        axes[idx].set_ylabel('Vérité', fontsize=12)
        axes[idx].set_xlabel('Prédiction', fontsize=12)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        text = f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}'
        axes[idx].text(1.5, 0.5, text, fontsize=10, 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Matrices de confusion par rôle (seuil = {threshold})', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Accuracy par label (moyenne)
    label_accuracy = (y_true == y_pred).mean(axis=0)
    print(f"\nAccuracy par label:")
    for tag, acc in zip(tags, label_accuracy):
        print(f"  {tag:12s}: {acc:.3f} ({acc*100:.1f}%)")
    
    # Analyse des confusions entre rôles
    n_classes = len(tags)
    confusion_add = np.zeros((n_classes, n_classes), dtype=int)  # Rôles ajoutés à tort
    confusion_miss = np.zeros((n_classes, n_classes), dtype=int)  # Rôles manqués
    
    for champion_idx in range(len(champion_names)):
        true_roles = set([j for j in range(n_classes) if y_true[champion_idx][j] == 1])
        pred_roles = set([j for j in range(n_classes) if y_pred[champion_idx][j] == 1])
        
        # Rôles prédits en trop (faux positifs)
        extra_roles = pred_roles - true_roles
        for extra in extra_roles:
            for true_role in true_roles:
                confusion_add[true_role][extra] += 1
        
        # Rôles manqués (faux négatifs)
        missing_roles = true_roles - pred_roles
        for missing in missing_roles:
            for pred_role in pred_roles:
                confusion_miss[missing][pred_role] += 1
    
    # Graphiques des confusions
    _ , (ax1, ax) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Rôles ajoutés à tort quand un rôle est vrai
    sns.heatmap(confusion_add, annot=True, fmt='d', cmap='Reds', 
                xticklabels=tags, yticklabels=tags, ax=ax1,
                cbar_kws={'label': 'Nombre d\'erreurs'})
    ax1.set_title('Rôles ajoutés à tort\n(ligne = vrai rôle, colonne = rôle ajouté)', 
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Rôle réel présent', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Rôle prédit en trop', fontsize=11, fontweight='bold')
    
    # Rôles manqués et remplacés par autre chose
    sns.heatmap(confusion_miss, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=tags, yticklabels=tags, ax=ax,
                cbar_kws={'label': 'Nombre d\'erreurs'})
    ax.set_title('Rôles manqués\n(ligne = vrai rôle manqué, colonne = rôle prédit à la place)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Rôle réel manqué', fontsize=11, fontweight='bold')
    ax.set_xlabel('Rôle prédit à la place', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    _ , ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Distribution par label
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    x = np.arange(len(tags))
    width = 0.35
    
    ax.bar(x - width/2, true_counts, width, label='Vérité', color='#2ca02c', alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label='Prédiction', color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Rôles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de champions', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des rôles', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return y_true, y_pred, champion_names