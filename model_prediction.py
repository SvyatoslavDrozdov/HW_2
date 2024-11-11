from functions import *

model = SegModel()

model.load_state_dict(torch.load('model_weights_20_epoches_07.pth', weights_only=True))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Evaluation started.")
time_1 = time.time()
evaluate(model, val_loader, device)
print(f"evaluation was made in {time.time() - time_1} sec.")
print("Writing submission started.")
write_submission(model, test_loader, device, thresholds)
