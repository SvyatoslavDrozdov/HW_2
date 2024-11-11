from functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, criterion, optimizer = init_model()
model.to(device)

print("Train started")
start_train_time = time.time()
train(model, criterion, optimizer, train_loader, epochs=20, my_device=device)
print(f"Train was made in {time.time() - start_train_time} seconds.")
print("Saving model")
torch.save(model.state_dict(), 'model_weights.pth')
