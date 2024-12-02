init:
	python main.py -w ./workspace gen-config

init-overwrite:
	python main.py -w ./workspace gen-config --overwrite

download-imagenet:
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

build-docs:
	sphinx-apidoc -o ./build-docs/source/docstring ./reliabilitycli;
	sphinx-build -b html ./build-docs/source ./docs;

#tensorboard:
#    command="tensorboard --logdir=runs --bind_all"
#    eval ${command}

clean:
	rm -rf dist reliabilitycli.egg-info


publish-test: clean
	python -m build
	twine upload dist/* --verbose --repository testpypi

publish: clean
	python -m build
	twine upload dist/*


.PHONY: build-docs