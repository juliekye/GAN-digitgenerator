from contextlib import suppress
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import cv2
import imageio
from models import Generator, Discriminator

plt.ion()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
n_epoch = 500
batch_size = 1000
noise_dim = 100
lr = 2e-4
snapshot_z = torch.randn(16, noise_dim).to(device)


def save(G, D):
    torch.save(G, 'model/generator.pkl')
    torch.save(D, 'model/discriminator.pkl')


def load():
    global G, D
    with suppress(Exception):
        G = torch.load('model/generator.pkl')

    with suppress(Exception):
        D = torch.load('model/discriminator.pkl')


# Prepare the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# build network
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim=noise_dim, g_output_dim=mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)
load()


# loss
criterion = nn.BCELoss()

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


def D_train(x):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = Variable(torch.randn(batch_size, noise_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(batch_size, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(batch_size, noise_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def show_progress(G, epoch):
    G.eval()  # Set the generator to evaluation mode
    n_images = 16
    nrows = 4
    ncols = 4
    padding = 2

    # Generate a batch of images
    generated_images = G(snapshot_z).detach().cpu().numpy()

    # Reshape and unnormalize the images
    generated_images = 0.5 * (generated_images.reshape(n_images, 28, 28) + 1)

    # Create a grid of images with padding
    grid_image = np.zeros((nrows * (28 + padding) - padding, ncols * (28 + padding) - padding), dtype=np.float32)

    for i in range(nrows):
        for j in range(ncols):
            img_idx = i * ncols + j
            grid_image[i * (28 + padding):(i + 1) * 28 + i * padding, j * (28 + padding):(j + 1) * 28 + j * padding] = generated_images[img_idx]

    # Show the grid of images using OpenCV
    window_title = 'Generated Images'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, grid_image)
    cv2.waitKey(1000)  # Display the window for 1 second

    # Save the generated images to a file
    snapshot_dir = 'snapshots2'
    os.makedirs(snapshot_dir, exist_ok=True)
    imageio.imsave(os.path.join(snapshot_dir, f'{epoch}.png'), (grid_image * 255).astype(np.uint8))

    G.train()  # Set the generator back to training mode


for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    show_progress(G, epoch)
    save(G, D)


print('Saving model...')

with torch.no_grad():
    test_z = Variable(torch.randn(batch_size, noise_dim).to(device))
    generated = G(test_z)

    save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + '.png')
