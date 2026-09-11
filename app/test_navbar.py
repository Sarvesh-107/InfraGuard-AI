"""Regression check for the shared top navbar (components/navbar.py::render_navbar).

Guards against the bug class reported once: the navbar (or a whole other page's
content) rendering twice on one page -- e.g. the shared component being invoked more
than once per page, or leftover markup from a different page's script not being
cleared between renders. Runs each page's actual script via Streamlit's own
AppTest harness (no browser needed) and asserts every nav item appears exactly once.

    python test_navbar.py
"""
from collections import Counter

from streamlit.testing.v1 import AppTest

from components.navbar import NAV_ITEMS

PAGES = [
    "Home.py",
    "pages/1_Early_Warning.py",
    "pages/2_Project_Detail.py",
    "pages/3_Models.py",
    "pages/4_Benchmarking.py",
    "pages/5_GIS_Map.py",
    "pages/6_Assistant.py",
]

# every nav tab label -- the full set render_navbar() puts on screen. No theme
# toggle: the app is light-mode only, with no "Dark Mode" button to expect.
EXPECTED_LABELS = [item["label"] for item in NAV_ITEMS]


def _navbar_button_counts(page_path):
    at = AppTest.from_file(page_path)
    at.run(timeout=120)
    assert not at.exception, f"{page_path} raised: {at.exception}"
    labels = [b.label for b in at.button if b.label in EXPECTED_LABELS]
    return Counter(labels)


def test_navbar_renders_exactly_once_per_page():
    for page in PAGES:
        counts = _navbar_button_counts(page)
        for label in EXPECTED_LABELS:
            assert counts[label] == 1, (
                f"{page}: expected exactly 1 {label!r} nav button, found "
                f"{counts[label]} -- navbar rendered more than once, or another "
                f"page's navbar leaked in"
            )
        print(f"ok  {page}: navbar renders exactly once "
              f"({len(EXPECTED_LABELS)} nav elements, no duplicates)")


if __name__ == "__main__":
    test_navbar_renders_exactly_once_per_page()
    print("\nall checks passed")
