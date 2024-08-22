import torch
import numpy as np

np.set_printoptions(linewidth=95)

with open("MCNN_parameters.pkl", 'rb') as f:
    file = torch.load(f)

    fc1_weight = file["fc1.weight"].cpu() * 100
    fc2_weight = file["fc2.weight"].cpu() * 100

    fc1_weight = fc1_weight.to(int)
    fc2_weight = fc2_weight.to(int)

    fc1_minus = fc1_weight < 0
    # fc2_minus = fc2_weight < 0

    fc1_weight[fc1_minus] = 16384 + fc1_weight[fc1_minus]
    # fc2_weight[fc2_minus] = 4096 + fc2_weight[fc2_minus]

    # print(fc1_weight[0].view(15, 15).flip(dims=[1]).numpy())
    # print("\n")
    # print(fc1_weight[1].view(15, 15).flip(dims=[1]).numpy())
    # print("\n")
    # print(fc1_weight[2].view(15, 15).flip(dims=[1]).numpy())
    # print("\n")
    # print(fc1_weight[3].view(15, 15).flip(dims=[1]).numpy())
    # print("\n")
    # print(fc2_weight)

    print(fc1_weight[0].view(15, 15).numpy())
    print("\n")
    print(fc1_weight[1].view(15, 15).numpy())
    print("\n")
    print(fc1_weight[2].view(15, 15).numpy())
    print("\n")
    print(fc1_weight[3].view(15, 15).numpy())
    print("\n")
    print(fc2_weight)
    print()


    def command(i, x, y, z):
        with open(f"./weights_command_{i}.mcfunction", "w") as f:
            w = fc1_weight[i].view(15, 15).flip(dims=[1]).numpy()
            for j in range(15):
                for k in range(15):
                    if i % 2 == 0:
                        f.write(f"data merge block {x + k * 2 + (1 - j % 2)} {y - (j // 2) * 4} {z} {{SuccessCount:{w[j][k]}}}\n")
                    else:
                        f.write(f"data merge block {x + k * 2 +      j % 2 } {y - (j // 2) * 4} {z} {{SuccessCount:{w[j][k]}}}\n")

    # command(0, 43, 49, 98)
    # command(1, 74, 49, 98)
    # command(2, 107, 49, 98)
    # command(3, 138, 49, 98)

