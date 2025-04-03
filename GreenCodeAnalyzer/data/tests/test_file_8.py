import torch
import torch.nn as nn
from transformers import BertModel
from transformers import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from TriggerModel.dataPipeline import TriggerModelPipeline
import json

class TriggerModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', numerical_feature_dim=3):

        super(TriggerModel, self).__init__()
        logging.set_verbosity(10)

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.num_feature_layer = nn.Sequential(
            nn.Linear(numerical_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.bert_hidden_size + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, numerical_features):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output

        numerical_output = self.num_feature_layer(numerical_features)

        combined_features = torch.cat((pooled_output, numerical_output), dim=1)
        output = self.fc(combined_features)

        return self.sigmoid(output)


class TriggerDataset(Dataset):
    def __init__(self, data_file, label_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        self.pipeline = TriggerModelPipeline()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        label = self.labels[idx]

        processed = self.pipeline.process_entry(entry)
        input_ids = processed["input_ids"].squeeze()
        attention_mask = processed["attention_mask"].squeeze()

        numerical_features = torch.tensor([
            entry["time_since_last_change"] / 1000.0,
            entry["line_count"],
            entry["char_amount"]
        ], dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)
        return input_ids, attention_mask, numerical_features, label

def train(dataAddressInput, dataAddressLabel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriggerModel().to(device)
    dataset = TriggerDataset(dataAddressInput, dataAddressLabel)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, numerical_features, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            numerical_features = numerical_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

def predict_with_rule_and_model(pipeline, model, entry):

    if pipeline.apply_rules(entry):
        #print("Rule triggered: Triggering completion.")
        return True

    processed = pipeline.process_entry(entry)
    input_ids = processed["input_ids"].unsqueeze(0)
    attention_mask = processed["attention_mask"].unsqueeze(0)
    numerical_features = torch.tensor([
        entry["time_since_last_change"] / 1000.0,
        entry["line_count"],
        entry["char_amount"]
    ], dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask, numerical_features).item()
        print(f"Model Output: {output:.4f}")
        return output > 0.5


if __name__ == "__main__":
    model = TriggerModel()