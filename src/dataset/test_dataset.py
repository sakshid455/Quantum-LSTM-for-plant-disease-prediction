from src.dataset.sequence_dataset import WheatSequenceDataset

dataset = WheatSequenceDataset()

print("Samples :", len(dataset))

X, y = dataset[0]

print()

print("Feature Shape :", X.shape)

print("Target :", y)
