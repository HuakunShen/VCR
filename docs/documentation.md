# Building Documentation

Our documentation is building with Sphinx, `sphinx_rtd_theme` theme. 

Python code documentation is generated with sphinx's autodoc extension.

Both rst and markdown format are supported.

## Writing and Building

1. `cd` into `doc` folder
2. Modify any `rst` and `md` files in this folder
3. If more code files are added, run `sphinx-apidoc -o code-doc ..` within the `doc` folder
3. `make html` to build the documentation in html
4. A `index.html` will be generated in `_build/html` folder, and that's the static file we want to host on the internet

## Deployment

There are too many ways to deploy a static file, we chose Netlify. 

To deploy the documentation manually, use `netlify-cli`.

`npm install netlify-cli -g` will install the tool, but in case you want to run it in a separated environment, I made a `Dockerfile` (`doc/netlify.Dockerfile`) for `netlify-cli`.

`docker build . -f ./netlify.Dockerfile -t netlify-cli` builds the image.

`docker run -it --rm netlify-cli bash` starts a interactive shell in the image.

`netlify-cli` supports multiple logins, run `netlify switch` to switch between them.

Run `netlify login` to login to your account.

Run `netlify deploy` and follow the instructions to upload the documentation.
    When it asks for publish directory, use `doc/_build/html`.

Use `netlify deploy --prod` for a production build, otherwise it will only be a preview page.
