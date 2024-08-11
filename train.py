
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses = []
    train_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_time = time.time()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        train_accuracy = calculate_accuracy(model, train_loader, device)
        train_accuracies.append(train_accuracy)
        scheduler.step()

        log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% ".format(epoch, running_loss, train_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)

    return train_losses, train_accuracies

criterion = torch.nn.CrossEntropyLoss()
model.to(device)
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Hyperparameters
batch_size = 1024
learning_rate = 0.001
epochs = 6
weight_decay = 1e-4

train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, scheduler, epochs, device)
