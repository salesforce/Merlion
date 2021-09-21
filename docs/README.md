To generate documentation using [Sphinx](https://www.sphinx-doc.org/en/master/index.html),  just run the script
[`build_docs.sh`](build_docs.sh). The ``build/html`` directory will be populated with searchable, 
indexed HTML documentation.

Note that our documentation also depends on [Pandoc](https://pandoc.org/installing.html) to render Jupyter notebooks.
For Ubuntu, call ``sudo apt-get install pandoc``. For Mac OS, install [Homebrew](https://brew.sh/)
and call ``brew install pandoc``.
