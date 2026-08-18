from src.dataset.multimodal_dataloader import (
    create_multimodal_loaders
)


train_loader, val_loader, test_loader, lesion_mean, lesion_std = \
    create_multimodal_loaders(batch_size=8)


X, y_disease, y_lesion = next(iter(train_loader))

print()
print("BATCH TEST")
print("=" * 40)
print("X shape        :", X.shape)
print("Disease shape  :", y_disease.shape)
print("Lesion shape   :", y_lesion.shape)