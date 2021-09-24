# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from git import Repo
import os
import pkg_resources
import re
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Merlion"
copyright = "2021, salesforce.com, inc."

# The full version, including alpha/beta/rc tags
release = pkg_resources.get_distribution("salesforce-merlion").version

default_role = "any"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["nbsphinx", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx_autodoc_typehints"]

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_options = {
    # Make sure that any autodoc declarations show the right members
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True  # Make _autosummary files and include them

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {"navigation_depth": -1}

# Set up something to display versions, but only do it if the current version is set in the environment.
if "current_version" in os.environ:
    current_version = os.environ["current_version"]
    stable_version = os.environ.get("stable_version", "latest")
    if current_version == stable_version != "latest":
        current_version = f"{current_version} (stable)"
    try:
        html_context
    except NameError:
        html_context = dict()
    html_context["display_lower_left"] = True

    repo = Repo(search_parent_directories=True)
    html_context["current_version"] = current_version
    html_context["version"] = current_version
    versions = sorted([tag.name for tag in repo.tags if re.match("v[0-9].*", tag.name)], reverse=True)
    versions = ["latest", *versions]
    html_context["versions"] = versions
