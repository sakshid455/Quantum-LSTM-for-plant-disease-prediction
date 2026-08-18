import pandas as pd
import matplotlib.pyplot as plt


def plot_loss():

    loss = pd.read_csv("outputs/loss_history.csv")

    plt.plot(
        loss["Epoch"],
        loss["Train Loss"],
        label="Train Loss"
    )

    plt.plot(
        loss["Epoch"],
        loss["Validation Loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("QLSTM Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_loss()