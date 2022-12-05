#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Script which removes redirects from the HTML API docs & updates the version matrix on old files.
"""
import os
import re
import shutil

from bs4 import BeautifulSoup as bs
from git import Repo


def create_version_dl(soup, prefix, current_version, all_versions):
    dl = soup.new_tag("dl")
    dt = soup.new_tag("dt")
    dt.string = "Versions"
    dl.append(dt)
    for version in all_versions:
        # Create the href for this version & bold it if it's the current version
        href = soup.new_tag("a", href=f"{prefix}/{version}/index.html")
        href.string = version
        if version == current_version:
            strong = soup.new_tag("strong")
            strong.append(href)
            href = strong
        # Create a list item & add it to the dl
        dd = soup.new_tag("dd")
        dd.append(href)
        dl.append(dd)
    return dl


def main():
    # Get all the versions
    repo = Repo(search_parent_directories=True)
    versions = sorted([tag.name for tag in repo.tags if re.match("v[0-9].*", tag.name)], reverse=True)
    versions = ["latest", *versions]

    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "html")
    for version in os.listdir(dirname):
        # If this isn't a directory containing a numbered version's API docs, delete it
        version_root = os.path.join(dirname, version)
        if version == "latest" or version not in versions:
            shutil.rmtree(version_root) if os.path.isdir(version_root) else os.remove(version_root)
            continue

        # Update version matrix in HTML source versioned files
        for subdir, _, files in os.walk(version_root):
            html_files = [os.path.join(subdir, f) for f in files if f.endswith(".html")]

            # Determine how far the version root is from the files in this directory
            prefix = ".."
            while subdir and subdir != version_root:
                subdir = os.path.dirname(subdir)
                prefix += "/.."

            # Create the new description list for the version & write the new file
            for file in html_files:
                with open(file) as f:
                    soup = bs(f, "html.parser")
                version_dl = [dl for dl in soup.find_all("dl") if dl.find("dt", text="Versions")]
                if len(version_dl) == 0:
                    continue
                version_dl[0].replace_with(create_version_dl(soup, prefix, version, versions))
                with open(file, "w", encoding="utf-8") as f:
                    f.write(str(soup))


if __name__ == "__main__":
    main()
