import torch
import numpy as np
import torch.utils
import torch.utils.data

if __name__ == "__main__":
    device = torch.device("cuda:0")

    a = np.zeros((400,3,224,224),dtype=float)
    b = np.zeros((400,),dtype=float)
    c = np.zeros((400,),dtype=float)

    dataset = torch.utils.data.TensorDataset(torch.Tensor(a),
                                            torch.Tensor(b),
                                            torch.Tensor(c)
                                            )

    dataLoader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            )

    for batch_idx, (x,y,y_id) in enumerate(dataLoader):

        x = x.to(device)
        y = y.to(device)
        y_id = y_id.to(device)

        for j in range(10000):
            y_new = y + y_id
            x_new = x*y_new
            x_new = x_new*x_new
