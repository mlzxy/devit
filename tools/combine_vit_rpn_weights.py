import torch
import os
import os.path as osp
import fire

root = osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__))))

model_dict = {
    's': osp.join(root, 'weights/initial/DINOv2/vits14.pth'),
    'b': osp.join(root, 'weights/initial/DINOv2/vitb14.pth'),
    'l': osp.join(root, 'weights/initial/DINOv2/vitl14.pth')
}

rpn_dict = {
    'coco': osp.join(root, 'weights/open-vocabulary-coco17/rpn_coco_48.pth'),
    'lvis': osp.join(root, 'weights/open-vocabulary-lvis/rpn_lvis_866_lsj.pth'),
    
    'os1': osp.join(root, 'weights/initial/rpn/one-shot-split1/model_final.pth'),
    'os2': osp.join(root, 'weights/initial/rpn/one-shot-split2/model_final.pth'),
    'os3': osp.join(root, 'weights/initial/rpn/one-shot-split3/model_final.pth'),
    'os4': osp.join(root, 'weights/initial/rpn/one-shot-split4/model_final.pth'),
    'fs14': osp.join(root, 'weights/initial/rpn/few-shot-coco14/model_final.pth'),  
}


def main(m='s', rpn='coco', out=None):
    if out is None:
        if rpn in ('lvis', 'coco'):
            out = osp.join(root, 'exp', m)
        else:
            out = osp.join(root, "pretrained_ckpt/xyz", rpn)

    if osp.isdir(out):
        out = osp.join(out, f'vit{m}+rpn{"_lvis" if rpn == "lvis" else ""}.pth')
        os.makedirs(osp.dirname(out), exist_ok=True)

    assert m in ['s', 'b', 'l']
    assert rpn in list(rpn_dict.keys())
    weights = {'model': torch.load(model_dict[m], map_location='cpu')}

    weights['model'] = {'backbone.' + k: v for k,v in weights['model'].items()}
    
    rpn_weights = torch.load(rpn_dict[rpn], map_location='cpu')

    for original_k in rpn_weights['model']:
        if 'backbone' in original_k:
            new_key = original_k.replace('backbone.', 'offline_backbone.')
            weights['model'][new_key] = rpn_weights['model'][original_k]
        elif 'proposal_generator' in original_k:
            new_key = original_k.replace('proposal_generator', 'offline_proposal_generator')
            weights['model'][new_key] = rpn_weights['model'][original_k]
        else:
            print(f'SKIP {original_k}')
    
    torch.save(weights, out)


if __name__ == "__main__":
    fire.Fire(main)