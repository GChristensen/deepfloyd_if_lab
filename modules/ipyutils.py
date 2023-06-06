import sys

def unload_modules(of):
    if_modules = []
    for n, m in sys.modules.items():
        if hasattr(m, "__file__") and m.__file__ and of in m.__file__:
            if_modules.append(n)

    for n in if_modules:
        sys.modules.pop(n)