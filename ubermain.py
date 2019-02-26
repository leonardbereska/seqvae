import subprocess

# p = subprocess.Popen('pwd', stdout=subprocess.PIPE)
# output, error = p.communicate()
# print(output)


name = 'sgd_vs_seqvae'
T = 100
n_epochs = 30000
lr = 2e-4
optims = ['sgd', 'seqvae']
dxdz = (5, 2)
# dxdzs = [(20, 5), (100, 10)]
times = 5
for i in range(times):
    saved_ = False
    for optim in optims:
        if saved_:
            load = '--load'
        else:
            saved_ = True
            load = ''

        dx = dxdz[0]
        dz = dxdz[1]

        comment = 'x{}z{}T{}{}'.format(dx, dz, T, optim)
        bash_command = 'python main.py --name {} {} --T {} --n_epochs {} ' \
                       '--lr {} --optim {} --dx {} --dz {} --comment {}'.format(name, load, T, n_epochs, lr, optim, dx, dz, comment)
        print(bash_command)
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()  # output, error


name = 'observe'
T = 100
n_epochs = 30000
lr = 2e-4
optims = ['sgd', 'seqvae']
dxdzs = [(2, 2), (5, 2), (20, 2)]
# dxdzs = [(20, 5), (100, 10)]

for dxdz in dxdzs:
    saved_ = False
    for optim in optims:
        if saved_:
            load = '--load'
        else:
            saved_ = True
            load = ''

        dx = dxdz[0]
        dz = dxdz[1]

        comment = 'x{}z{}T{}{}'.format(dx, dz, T, optim)
        bash_command = 'python main.py --name {} {} --T {} --n_epochs {} ' \
                       '--lr {} --optim {} --dx {} --dz {} --comment {}'.format(name, load, T, n_epochs, lr, optim, dx, dz, comment)
        print(bash_command)
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()  # output, error


name = 'states'
T = 100
n_epochs = 30000
lr = 2e-4
optims = ['sgd', 'seqvae']
dxdzs = [(5, 2), (10, 5), (20, 10)]
# dxdz = (5, 2)

for dxdz in dxdzs:
    saved_ = False
    for optim in optims:
        if saved_:
            load = '--load'
        else:
            saved_ = True
            load = ''

        dx = dxdz[0]
        dz = dxdz[1]

        comment = 'x{}z{}{}'.format(dx, dz, T, optim)
        bash_command = 'python main.py --name {} {} --T {} --n_epochs {} ' \
                       '--lr {} --optim {} --dx {} --dz {} --comment {}'.format(name, load, T, n_epochs, lr, optim, dx, dz, comment)
        print(bash_command)
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()


name = 'times'
Ts = [100, 1000, 10000]
n_epochs = 30000
lr = 2e-4
optims = ['sgd', 'seqvae']
dxdz = (5, 2)
# dxdzs = [(20, 5), (100, 10)]

for T in Ts:
    saved_ = False
    for optim in optims:
        if saved_:
            load = '--load'
        else:
            saved_ = True
            load = ''

        dx = dxdz[0]
        dz = dxdz[1]

        comment = 'x{}z{}T{}{}'.format(dx, dz, T, optim)
        bash_command = 'python main.py --name {} {} --T {} --n_epochs {} ' \
                       '--lr {} --optim {} --dx {} --dz {} --comment {}'.format(name, load, T, n_epochs, lr, optim, dx, dz, comment)
        print(bash_command)
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

