import importlib.util

mod = importlib.util.spec_from_file_location('model', './workspace/model.py').loader.load_module()

print(mod.model)
print(hasattr(mod, 'model'))
print(hasattr(mod, 'model2'))

