# Comfyui lama remover
A very simple ComfyUI node to remove item with mask.

## Install
- Navigate to `/ComfyUI/custom_nodes/` folder 
- Run \
`git clone https://github.com/Layer-norm/comfyui-lama-remover.git` 
- Restart ComfyUI 
- (If downlaod error occued) \
Download [big-lama.pt](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) model to `/ComfyUI/custom_nodes/comfyui-lama-remover/ckpts/` folder

## Examples
![remove with mask](example\example1.png)
![remove with mask(img)](example\example2.png)

## Acknowledgments
[ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) The most powerful and modular stable diffusion GUI and backend. \
[lama](https://github.com/advimman/lama) 🦙 LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions.[[1]](https://arxiv.org/abs/2109.07161)  \
[simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) Simple pip package for LaMa inpainting.\
[lama-cleaner](https://github.com/Sanster/lama-cleaner) A free and open-source inpainting tool powered by SOTA AI model.
