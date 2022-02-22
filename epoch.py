import os
import pandas as pd
from tqdm.auto import tqdm
import torch

best_acc = 0
best_epoch_idx = None
epoch_patience = 0

def train_epoch(args, epoch_idx, model, dataloader, optimizer, loss_fn, writer, device):
    model = model.train()

    epoch_acc = 0
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f'TRAIN EPOCH {epoch_idx}/{args.epoch}')):
        input = batch_data['text'].to(device)
        target = batch_data['label'].squeeze().to(device)

        outputs = model(input)
        loss = loss_fn(outputs, target)

        batch_acc = torch.sum(torch.argmax(outputs, dim=1) == target).item() / target.size(0) * 100
        epoch_acc += batch_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
            tqdm.write(f'TRAIN: {batch_idx}/{len(dataloader)} - Loss={loss.item()} Acc={batch_acc}')
        if args.use_tensorboard:
            total_idx = batch_idx + (epoch_idx * len(dataloader))
            writer.add_scalar('TRAIN/Loss', loss.item(), total_idx)
            writer.add_scalar('TRAIN/Batch_Accuracy', batch_acc, total_idx)
        
    epoch_acc /= len(dataloader)
    if args.use_tensorboard:
        writer.add_scalar('TRAIN/Epoch_Accuracy', epoch_acc, epoch_idx)

def valid_epoch(args, epoch_idx, model, dataloader, optimizer, loss_fn, writer, device):
    model = model.eval()

    epoch_acc = 0
    global best_acc
    global best_epoch_idx
    global epoch_patience
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f'VALID EPOCH {epoch_idx}/{args.epoch}')):
        input = batch_data['text'].to(device)
        target = batch_data['label'].squeeze().to(device)

        with torch.no_grad():
            outputs = model(input)
            loss = loss_fn(outputs, target)

        batch_acc = torch.sum(torch.argmax(outputs, dim=1) == target).item() / target.size(0) * 100
        epoch_acc += batch_acc

        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
            tqdm.write(f'VALID: {batch_idx}/{len(dataloader)} - Loss={loss.item()} Acc={batch_acc}')
        if args.use_tensorboard:
            total_idx = batch_idx + (epoch_idx * len(dataloader))
            writer.add_scalar('VALID/Loss', loss.item(), total_idx)
            writer.add_scalar('VALID/Batch_Accuracy', batch_acc, total_idx)
        
    epoch_acc /= len(dataloader)
    if args.use_tensorboard:
        writer.add_scalar('VALID/Epoch_Accuracy', epoch_acc, epoch_idx)

    # Save the model if the validation accuracy is the best we've seen so far.
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch_idx = epoch_idx
        epoch_patience = 0
        
        save_model_name = f'{args.model_name}_{best_epoch_idx}.pt'
        torch.save({
            'epoch': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.save_model_path, save_model_name))
    else:
        epoch_patience += 1
        if args.early_stopping_patience is not None and epoch_patience >= args.early_stopping_patience:
            tqdm.write(f'Early Stopping at Epoch {epoch_idx}')
            return False

def test_model(args, model, dataloader, loss_fn, writer, device):
    global best_acc
    global best_epoch_idx
    global epoch_patience
    # load the best model from the validation set
    load_model_name = f'{args.model_name}_{best_epoch_idx}.pt'
    checkpoint = torch.load(os.path.join(args.save_model_path, load_model_name))
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    tqdm.write(f'Loaded Best Model: {load_model_name}')

    total_acc = 0
    reference_text = []
    generated_label = []
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f'TEST SEQUENCE')):
        input = batch_data['text'].to(device)
        target = batch_data['label'].squeeze().to(device)

        with torch.no_grad():
            outputs = model(input)
            loss = loss_fn(outputs, target)

        batch_acc = torch.sum(torch.argmax(outputs, dim=1) == target).item() / target.size(0) * 100
        total_acc += batch_acc

        for i in range(input.size(0)):
            reference_text.append(batch_data['text'][i])
            generated_label.append(torch.argmax(outputs, dim=1)[i].cpu())

        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
            tqdm.write(f'TEST: {batch_idx}/{len(dataloader)} - Loss={loss.item()} Acc={batch_acc}')
        if args.use_tensorboard:
            writer.add_scalar('TEST/Loss', loss.item(), batch_idx)
            writer.add_scalar('TEST/Batch_Accuracy', batch_acc, batch_idx)

    total_acc /= len(dataloader)

    tqdm.write("Complete Model Test Sequence - Accuracy: {}".format(total_acc))
    if args.use_tensorboard:
        writer.add_text('TEST/Total_Accuracy', str(total_acc))

    # reset global variables for grid search
    best_acc = 0
    best_epoch_idx = None
    epoch_patience = 0
