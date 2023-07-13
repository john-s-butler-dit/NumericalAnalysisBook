<!-- #region -->
# Jupyterbook Support

We use [jupyterbook](https://jupyterbook.org/intro.html) to create a condensed,
single interface to all our course materials. This book will be autogenerated
and published to [https://github.com/john-s-butler-dit/Numerical-Analysis-Python] every time
a PR is merged into the master branch.

In order to build the book locally, you will need to do the following:

1. Install jupyter-book 

`pip install jupyter-book`


2. Clean the book


`jupyter-book clean BOOK/'

3. First build of the book

`jupyter-book build --all BOOK/`

4. Subsequent builds of the book

`jupyter-book build BOOK/`

5. Now copy contents to webpage folder

`cp -R BOOK/ NumericalAnalysisBook/`

6. Install github pages

`pip install ghp-import`


7. Go to NumericalAnalysisBook folder


8. Push to GIT

`ghp-import -n -p -f _build/html`

Now enjoy:
[https://john-s-butler-dit.github.io/NumericalAnalysisBook/]
<!-- #endregion -->

```python

```