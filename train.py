import argparse
import os
import numpy as np
from dataloader import cityDataset
from models.generator import generator_model
from models.discriminator import discriminator_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision.utils import make_grid , save_image


def arguments():
    parser = argparse.ArgumentParser(description = "Define dynamic parameters for the model.")
    parser.add_argument('--epochs' , default = 20 , type = int , help = 'number of epochs')
    parser.add_argument('--train_data_dir' , default = r'city_data\train' , type = str , help = 'training data directory path')
    parser.add_argument('--val_data_dir' , default = r'city_data\val' , type = str , help = 'validation data directory path')
    parser.add_argument('--train_batch_size' , default = 4 , type = int , help = 'define training data batch size')
    parser.add_argument('--val_batch_size' , default = 1 , type = int , help = 'define validation data batch size')
    parser.add_argument("--latent_dims", default=100, type = int , help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--learning_rate", default=0.0002, type = float , help="learning rate")
    parser.add_argument("--L1_loss_factor", default=100, type = int , help="define L1 loss factor")
    parser.add_argument("--beta1", default=0.5, type = float , help="value of beta1")
    parser.add_argument("--weight_dir", default="./weight", type = str , help="weight directory path")
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or cpu")
    return parser.parse_args() 

args = arguments()

def run():
    
    train_loader = DataLoader(dataset=cityDataset(data_dir=args.train_data_dir),batch_size=args.train_batch_size,shuffle=True)
    val_loader =  DataLoader(dataset=cityDataset(data_dir=args.val_data_dir,train=False),batch_size=args.val_batch_size)
    
    
    
    gen_model = generator_model().to(args.device)
    disc_model = discriminator_model().to(args.device)
   
    criterion_bce = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    
    opt_gen = torch.optim.Adam(gen_model.parameters(),lr = args.learning_rate,betas=(args.beta1,0.999))
    opt_disc = torch.optim.Adam(disc_model.parameters(),lr = args.learning_rate,betas=(args.beta1,0.999))
    
    
    

    # create folders
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs('./result' , exist_ok=True)

    ## training loop

    
    
    
    for epoch in range(args.epochs):
        gen_model.train(True)
        disc_model.train(True)
        total_loss_disc_train = 0.0
        total_loss_gen_train = 0.0
        last_loss_disc_train = 0.0
        last_loss_gen_train = 0.0
        total_disc_loss_val = 0.0 
        gen_loss_val = 0.0
        
        for idx,(batch_train_input,batch_train_target) in enumerate(train_loader):
            
            batch_train_input = batch_train_input.to(args.device)
            batch_train_target = batch_train_target.to(args.device)
            fake_img = gen_model(batch_train_input)
            fake_prediction = disc_model(torch.cat((batch_train_input,fake_img),dim=1))
            real_prediction = disc_model(torch.cat((batch_train_input,batch_train_target),dim=1))
            
            loss_real = criterion_bce(real_prediction , torch.ones_like(real_prediction))
            loss_fake = criterion_bce(fake_prediction , torch.zeros_like(fake_prediction))
            total_disc_loss = (loss_fake + loss_real) / 2
            
            disc_model.zero_grad()
            total_disc_loss.backward(retain_graph = True)
            opt_disc.step()
            
            fake_pred = disc_model(torch.cat((batch_train_input,fake_img),dim=1))
            gen_loss_bce = criterion_bce(fake_pred, torch.ones_like(fake_pred))
            gen_loss_l1 = criterion_L1(fake_img,batch_train_target)
            gen_loss = gen_loss_bce + (args.L1_loss_factor * gen_loss_l1)
            gen_model.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            
            total_loss_disc_train += total_disc_loss.item()
            total_loss_gen_train += gen_loss.item()
            
            if idx % 100 == 99:
                last_loss_disc_train = total_loss_disc_train / 100
                last_loss_gen_train = total_loss_gen_train / 100
                total_loss_disc_train = 0.0
                total_loss_gen_train = 0.0
                print('batch {} disc_loss: {} gen_loss: {}'.format(idx + 1, last_loss_disc_train , last_loss_gen_train))
            
        if epoch % 5 == 0:
            torch.save(disc_model.state_dict() , os.path.join(args.weight_dir , f'discriminator_ckpt_{epoch}.pt'))
            torch.save(gen_model.state_dict() , os.path.join(args.weight_dir , f'generator_ckpt_{epoch}.pt'))
        disc_model.eval()
        gen_model.eval()
        with torch.no_grad():
            batch_val_input = next(iter(val_loader))[0].to(args.device)
            generated_img = gen_model(batch_val_input)
            grid= make_grid(generated_img,normalize=True)
            save_image(grid , f'./result/generated_epoch{epoch}.jpg')
            
            
       
            
    
    
if __name__ == '__main__':
    
    run()