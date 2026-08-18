from src.dataset.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

x, y = next(iter(train_loader))

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)