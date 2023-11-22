import itertools


def make_attr(attr: str, idx: str):
    catlist = [idx.index(c) for c in attr]
    print(attr + ': "GraphicsNDArray"', "=", f"qattr((..., {catlist}))", file=fo)


indices = ['xyzw', 'rgba']
with open("caliborn/Q.py", "w") as fo:
    for idx in indices:
        comb = [''.join(x) for x in itertools.product([''] + list(idx), repeat=len(idx))]
        comb = list(dict.fromkeys(x for x in comb if x))
        for attr in comb:
            make_attr(attr, idx)
