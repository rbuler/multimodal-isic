import torch
import copy
from sklearn.metrics import classification_report, balanced_accuracy_score


def train(model, dataloader, criterion, optimizer, device, neptune_run, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        image, radiomics_feat, age, sex, loc, artifacts = (
            batch['image'].to(device),
            batch['radiomics'].to(device),
            batch['age'].to(device),
            batch['sex'].to(device),
            batch['loc'].to(device),
            batch['artifacts'].to(device)
        )
        target = batch['target'].to(device)
        for param in model.parameters():
            param.grad = None
        outputs = model(image, radiomics_feat, age, sex, loc, artifacts)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == target).sum().item()

        total += target.size(0)
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    if neptune_run is not None:
        neptune_run["train/epoch_loss"].log(epoch_loss)
        neptune_run["train/epoch_acc"].log(epoch_acc)
        if hasattr(model, "weights"):
            fusion_weights = model.weights.detach().cpu().numpy().tolist()
            for i, w in enumerate(fusion_weights):
                neptune_run[f"model/fusion_weight_modality_{i}"].log(w)
    print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


def validate(model, dataloader, criterion, device, neptune_run, epoch, fold_idx=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            image, radiomics_feat, age, sex, loc, artifacts = (
                batch['image'].to(device),
                batch['radiomics'].to(device),
                batch['age'].to(device),
                batch['sex'].to(device),
                batch['loc'].to(device),
                batch['artifacts'].to(device)
            )
            target = batch['target'].to(device)
            outputs = model(image, radiomics_feat, age, sex, loc, artifacts)
            loss = criterion(outputs, target)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["val/epoch_loss"].log(epoch_loss)
            neptune_run["val/epoch_acc"].log(epoch_acc)
    else:
        if neptune_run is not None:
            neptune_run[f"{fold_idx}/val/epoch_loss"].log(epoch_loss)
            neptune_run[f"{fold_idx}/val/epoch_acc"].log(epoch_acc)
        
    print(f"Epoch {epoch} - Val Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss


def test(model, dataloader, device, neptune_run, fold_idx=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            image, radiomics_feat, age, sex, loc, artifacts = (
                batch['image'].to(device),
                batch['radiomics'].to(device),
                batch['age'].to(device),
                batch['sex'].to(device),
                batch['loc'].to(device),
                batch['artifacts'].to(device)
            )
            target = batch['target'].to(device)
            outputs = model(image, radiomics_feat, age, sex, loc, artifacts)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_acc = correct / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["test/accuracy"].log(test_acc)
            neptune_run["test/classification_report"].log(report)
            neptune_run["test/balanced_accuracy"].log(balanced_acc)

    else:
        if neptune_run is not None:
            neptune_run[f"{fold_idx}/test/accuracy"].log(test_acc)
            neptune_run[f"{fold_idx}/test/classification_report"].log(report)
            neptune_run[f"{fold_idx}/test/balanced_accuracy"].log(balanced_acc)

    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:\n", report)
    return test_acc, report


class EarlyStopping:
    def __init__(self, patience=5, neptune_run=None):
        self.patience = patience
        self.counter = patience
        self.best_loss = float('inf')
        self.best_model_state = None
        self.neptune_run = neptune_run

    def __call__(self, current_loss, model):
        copy_model = False
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = self.patience
            copy_model = True
        else:
            self.counter -= 1

        if self.neptune_run is not None:
            self.neptune_run["val/patience_counter"].log(self.counter)

        if copy_model:
            self.best_model_state = copy.deepcopy(model.state_dict())

        return not self.counter

    def get_best_model_state(self):
        """Return the best model state dictionary."""
        return self.best_model_state