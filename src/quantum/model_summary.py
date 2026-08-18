from src.quantum.qlstm_model import QLSTMModel

model = QLSTMModel()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 40)
print("QLSTM MODEL SUMMARY")
print("=" * 40)

print(model)

print("\nTotal Parameters :", total)
print("Trainable Parameters :", trainable)