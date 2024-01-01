import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.init as init
from pyDOE import lhs
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('dark_background')
print(device.__repr__())

A_n = 1
n = 4
k = 0.02
L = 1

x_pts = 200
t_pts = 200

x_start = 0
t_start = 0

x_end = 1
t_end = 1

def f(x):
    return A_n * torch.sin((n * torch.pi * x) / L)

def u(x, t):
    return f(x) * torch.exp(-k * ((n * torch.pi / L) ** 2) * t)

x = torch.linspace(0, x_end, x_pts)
t = torch.linspace(0, t_end, t_pts)

x, t = torch.meshgrid(x, t)

def ic():
    rand_pts = lhs(x_pts, 1)[0]
    x = (rand_pts * (x_end - x_start)) + x_start
    t = np.zeros(1)
    x, t = np.meshgrid(x, t)
    pts = np.array(list(zip(x.ravel(), t.ravel())))
    return pts

def left_bc():
    rand_pts = lhs(t_pts, 1)[0]
    x = np.zeros(1)
    t = (rand_pts * (t_end - t_start)) + t_start
    x, t = np.meshgrid(x, t)
    pts = np.array(list(zip(x.ravel(), t.ravel())))
    return pts

def right_bc():
    rand_pts = lhs(t_pts, 1)[0]
    x = np.array(x_end)
    t = (rand_pts * (t_end - t_start)) + t_start
    x, t = np.meshgrid(x, t)
    pts = np.array(list(zip(x.ravel(), t.ravel())))
    return pts

def interior():
    rand_pts = lhs(x_pts, 1)[0]
    x = (rand_pts * (x_end - x_start)) + x_start
    rand_pts = lhs(t_pts, 1)[0]
    t = (rand_pts * (t_end - t_start)) + t_start
    x, t = np.meshgrid(x, t)
    pts = np.array(list(zip(x.ravel(), t.ravel())))
    return pts

ic_pts = torch.tensor(ic()).to(device, dtype=torch.float32)
left_bc_pts = torch.tensor(left_bc()).to(device, dtype=torch.float32)
right_bc_pts = torch.tensor(right_bc()).to(device, dtype=torch.float32)
interior_pts = torch.tensor(interior()).to(device, dtype=torch.float32)

bc_pts = torch.vstack((left_bc_pts, right_bc_pts, ic_pts))

bc_pts = bc_pts[torch.randperm(bc_pts.size(0))]
interior_pts = interior_pts[torch.randperm(interior_pts.size(0))]

print(f'Boundary Points: {bc_pts.shape, bc_pts.dtype}')
print(f'Interior Points: {interior_pts.shape, interior_pts.dtype}')

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        neurons = 64
        self.loss = nn.MSELoss(reduction='sum')
        self.network = nn.Sequential(
            nn.Linear(2, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 1)
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, input):
        return self.network(input)
    
    def bc_loss(self, pts):
        g = pts.clone()
        g.requires_grad=True
        pred = self.forward(g)
        '''
            [:, 0] = [0, 1, 2, ...]
            [:, [0]] = [[0], [1], [2], ...]
        '''
        y = u(g[:, 0], g[:, 1]).reshape(-1, 1)
        loss = self.loss(pred, y)
        return loss

    def phys_loss(self, pts):
        g = pts.clone()
        g.requires_grad=True
        pred = self.forward(g)
        '''
            [:, 0] = [0, 1, 2, ...]
            [:, [0]] = [[0], [1], [2], ...]
        '''
        first_order = autograd.grad(pred, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        dNN_dt = first_order[:, [1]]
        second_order = autograd.grad(first_order, g, torch.ones(g.shape).to(device), retain_graph=True, create_graph=True)[0]
        d2NN_dx2 = second_order[:, [0]]
        PDE = dNN_dt - (k * d2NN_dx2)
        '''
            F.mse_loss(PDE, 0) should be F.mse_loss(PDE, torch.zeros(PDE.shape[0],1))
            must be [[0], [0], [0], ...] not [0, 0, 0, ...]
        '''
        zeros = torch.zeros(PDE.shape[0], 1).to(device)
        loss = self.loss(PDE, zeros)
        return loss

    def total_loss(self, bc_pts, interior_pts):
        bc_loss = self.bc_loss(bc_pts)
        phys_loss = self.phys_loss(interior_pts)

        total_loss = bc_loss + phys_loss
        return bc_loss, phys_loss, total_loss
    
# helper function to format losses when printing
def format_number(num, length=21):
    num_str = str(num)

    if "." in num_str:
        integer_part, decimal_part = num_str.split(".")
        formatted_decimal_part = decimal_part.ljust(3, '0')
        formatted_integer_part = integer_part
        formatted_num = "{}.{}".format(formatted_integer_part, formatted_decimal_part)
    else:
        formatted_num = num_str

    formatted_num = formatted_num.ljust(length, '0')
    return formatted_num

model = PINN().to(device, dtype=torch.float32)
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-4)
epochs = 500

bc_losses = []
phys_losses = []
total_losses = []

def closure():
    optimizer.zero_grad()
    bc_loss, phys_loss, total_loss = model.total_loss(bc_pts, interior_pts)
    total_loss.backward()

    bc_losses.append(bc_loss.cpu().detach())
    phys_losses.append(phys_loss.cpu().detach())
    total_losses.append(total_loss.cpu().detach())

    bc_loss = format_number(bc_loss.item())
    phys_loss = format_number(phys_loss.item())
    total_loss = format_number(total_loss.item())

    print(f'-------------- BC Loss: {bc_loss} | Phys Loss: {phys_loss} | Total Loss: {total_loss}')

    return total_loss

torch.cuda.empty_cache()
model.train()
for i in range(1, epochs + 1):
    epoch = str(i).rjust(7)
    print(f'Epoch: {epoch}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(closure)
    scheduler.step(closure())

def save_model(file_path, epoch, model, optimizer, scheduler, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def load_model():
    file_path = 'model.tar'
    model = PINN().to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
    checkpoint = torch.load(file_path, map_location=device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-4)
    epoch = int(checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    loss = checkpoint['loss']

    return epoch, model, optimizer, scheduler, loss

save_model('model.tar', epoch, model, optimizer, scheduler, total_losses[-1])
# i, model, optimizer, scheduler, loss = load_model()

x = torch.linspace(x_start, x_end, x_pts)
t = torch.linspace(t_start, t_end, t_pts)

x, t = torch.meshgrid(x, t)
test = np.array(list(zip(x.ravel(), t.ravel())))
test = torch.tensor(test).to(device, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred = model(test)

pred = pred.reshape(t_pts, x_pts).to('cpu')

fig = plt.figure(figsize=(25, 30))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[1, 1])

axes1 = fig.add_subplot(gs[0, :])
axes2 = fig.add_subplot(gs[1, 0], projection='3d')
axes3 = fig.add_subplot(gs[1, 1], projection='3d')
axes4 = fig.add_subplot(gs[2, 0])
axes5 = fig.add_subplot(gs[2, 1])

axes4.set_aspect('equal')
axes5.set_aspect('equal')

contour = True

axes1.plot(bc_losses, label='Boundary', color='#9933FF')
axes1.plot(phys_losses, label='Physics', color='#3366FF')
axes1.plot(total_losses, label='Total Loss', color='orange')
axes1.set_title('Loss Plot')
axes1.set_xlabel('Epochs')
axes1.set_ylabel('Loss')
axes1.grid(True)
axes1.legend()

axes2.set_xlabel('x')
axes2.set_ylabel('t')
axes2.set_zlabel('NN(x, t)')
axes2.set_title('Prediction')
axes2.plot_surface(x, t, pred, cmap='viridis')

axes3.set_xlabel('x')
axes3.set_ylabel('t')
axes3.set_zlabel('u(x, t)')
axes3.set_title('Ground Truth')
axes3.plot_surface(x, t, u(x, t), cmap='viridis')

if contour == True:
    axes4.set_xlabel('x')
    axes4.set_ylabel('t')
    axes4.set_title('Prediction')
    axes4.contour(x, t, pred, 50)

    axes5.set_xlabel('x')
    axes5.set_ylabel('t')
    axes5.set_title('Ground Truth')
    axes5.contour(x, t, u(x, t), 50)

else:
    axes4.set_xlabel('x')
    axes4.set_ylabel('t')
    axes4.set_title('Prediction')
    axes4.contourf(x, t, pred, 50)

    axes5.set_xlabel('x')
    axes5.set_ylabel('t')
    axes5.set_title('Ground Truth')
    axes5.contourf(x, t, u(x, t), 50)

plt.subplots_adjust(hspace=0.5)

plt.show()