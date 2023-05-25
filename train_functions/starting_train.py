import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )


    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()


    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            print(type(batch)) #testing step to see batch type
            print("--------------------------------------------------") #separate
            images, labels = batch
            outputs = model(images)
            # TODO: Backpropagation and gradient descent
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch:', epoch, 'Loss:', loss.item())

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                tra_acc = compute_accuracy(outputs, labels)
                print("training accuracy is : ", tra_acc)
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                val_loss, val_acc = evaluate(val_loader, model, loss_fn)
                print("validation loss is : ", val_loss, " and accuracy is : ", val_acc)

            model.train()
            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    #predictions = torch.argmax(outputs, dim =1)
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    
    for the loss function, the detailed implementation isn't important,
    just make it consistent throughout the training. Just want the loss
    curve to go down in general. 
    But also look out for patterns of overfitting (validation loss curve 
    up), it needs to stop then.
    """
    model.eval()
    count, loss, correct = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = batch
            outputs = model(images)

            loss += loss_fn(outputs, labels).mean().item()
            count += len(labels)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct / count
    return loss, accuracy
    
    
