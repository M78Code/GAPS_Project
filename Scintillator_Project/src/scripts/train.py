import torch
import torch.nn as nn
from pathlib import Path
import sys

from torch.optim import Adam

sys.path.append(str(Path(__file__).parent))


from ..data_parse.data_loader import make_dataloaders
from ..models.cnn1d import  CNN1D

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'使用设备：{device}')
# 数据分割后的路径
split_dir = '../dataset/split'

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def train():
    # DataLoader
    split_dir = Path(__file__).parent.parent / 'dataset' / 'split'
    train_loader, val_loader, _ = make_dataloaders(split_dir, batch_size=BATCH_SIZE)

    # 模型
    model = CNN1D().to(device)

    # 损失函数 + 优化器
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)

    # 保存最优模型
    save_path = Path(__file__).parent.parent / 'results' / 'best_model.pth'
    save_path.parent.mkdir(exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ── 训练 ──────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)   # [batch] -> [batch, 1]

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x)

        train_loss /= len(train_loader.dataset)

        # ── 验证 ──────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                pred = model(x)
                loss = criterion(pred, y)
                
                val_loss += loss.item() * len(x)

        val_loss /= len(val_loader.dataset)

        # ── 保存最优 ───────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            mark = " ← 保存"
        else:
            mark = ""

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f}{mark}")

    print(f"\n训练完成。最优 val_loss: {best_val_loss:.4f}")
    print(f"模型保存于: {save_path}")


if __name__ == "__main__":
    train()
