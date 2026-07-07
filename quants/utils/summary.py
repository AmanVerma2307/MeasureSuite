import torch

def print_model_summary(model, input_size):

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "nb_params": sum(p.numel() for p in module.parameters())
            }
        if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)
    with torch.no_grad():
        model(torch.zeros(1, *input_size))

    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20} {:>15}".format("Layer (type)", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    for layer in summary:
        line_new = "{:>20} {:>15}".format(
            layer,
            "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print("----------------------------------------------------------------")
