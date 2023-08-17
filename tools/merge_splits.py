import sys

def torch_load(f):
    print(f'Loading {f}')
    return torch.load(f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('nothing to merge')
        sys.exit()
    import torch
    files = sys.argv[1:]

    parts = files[0].split('.')[:-1] 
    parts = [p for p in parts if 'i0' not in p]
    parts += ['merge.pkl']
    output_p = '.'.join(parts)

    print(f'Merging: {files}, output to {output_p}')
    data = [torch_load(f) for f in files]

    ks = data[0].keys()
    ks = [k for k in ks if isinstance(data[0][k], list)]
    print(f'Joining data: {ks} ...')
    dct = {
        k: sum([d[k] for d in data], [])
        for k in ks
    }

    print(f'Saving to {output_p}')
    torch.save(dct, output_p)