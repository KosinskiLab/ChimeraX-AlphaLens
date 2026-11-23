'''
The help pages installed with the bundle (src/docs/user/tools/, made by scripts/build_help.py):
current with their Markdown, and every link and image inside the bundle resolves.
'''
import hashlib
import html
import importlib.util
import os
import re
import tomllib

import pytest

from alphalens import helpdocs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HELP = os.path.join(ROOT, 'src', 'docs', 'user', 'tools')

spec = importlib.util.spec_from_file_location('build_help', os.path.join(ROOT, 'scripts', 'build_help.py'))
build_help = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_help)


def page_text(page):
    with open(os.path.join(HELP, page), encoding='utf-8') as f:
        return f.read()


@pytest.mark.parametrize('source, page, label', build_help.PAGES)
def test_page_is_current_with_its_markdown(source, page, label):
    m = build_help.HEADER.match(page_text(page))
    assert m is not None, f'{page} lacks the generated-page header'
    with open(os.path.join(ROOT, source), 'rb') as f:
        current = hashlib.sha256(f.read()).hexdigest()
    assert m.group(1) == source
    assert m.group(2) == current, (f'{page} is older than {source}: run '
                                   f'"python3 scripts/build_help.py" (see its docstring)')


def test_links_and_images_inside_the_bundle_resolve():
    pages = {page for _, page, _ in build_help.PAGES}
    ids = {page: set(re.findall(r'\sid="([^"]+)"', page_text(page))) for page in pages}
    problems = []
    for page in pages:
        for url in re.findall(r'\s(?:href|src)="([^"]*)"', page_text(page)):
            url = html.unescape(url)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', url):
                continue                    # Full URL
            path, _, fragment = url.partition('#')
            if path.startswith(build_help.IMAGES + '/'):
                image = os.path.join(ROOT, 'docs', 'images', path[len(build_help.IMAGES) + 1:])
                if not os.path.exists(image):
                    problems.append(f'{page}: image {url}')
                continue
            target = path or page
            if target not in pages:
                problems.append(f'{page}: link {url} to a page that is not made')
            elif fragment and fragment not in ids[target]:
                problems.append(f'{page}: link {url} to a missing heading')
    assert not problems, '\n'.join(problems)


def test_rewrite_url():
    r = build_help.rewrite_url
    assert r('docs/reference.md#commands', 'README.md') == 'alphalens-reference.html#commands'
    assert r('../README.md', 'docs/reference.md') == 'alphalens.html'
    assert r('local-prediction.md', 'docs/reference.md') == 'alphalens-prediction.html'
    assert r('docs/images/views.png', 'README.md') == 'alphalens_images/views.png'
    assert r('#setup', 'docs/local-prediction.md') == '#setup'
    assert r('https://example.org/a.md', 'README.md') == 'https://example.org/a.md'
    assert r('LICENSE', 'README.md') == build_help.GITHUB + 'blob/main/LICENSE'
    assert r('local-prediction-benchmark.json', 'docs/local-prediction.md') == \
        build_help.GITHUB + 'blob/main/docs/local-prediction-benchmark.json'
    assert r('../tests', 'docs/reference.md') == build_help.GITHUB + 'tree/main/tests'


def test_github_slug_matches_github():
    s = build_help.github_slug
    assert s('Getting a distogram') == 'getting-a-distogram'
    assert s('Is it the same interface?') == 'is-it-the-same-interface'
    assert s('What is verified, and what is not') == 'what-is-verified-and-what-is-not'


def test_help_module_names_the_generated_pages():
    made = {page: source for source, page, _ in build_help.PAGES}
    assert helpdocs.SOURCES == made
    assert {helpdocs.PLOT, helpdocs.REFERENCE, helpdocs.PREDICTION} == set(made)


def test_bundle_ships_the_pages_and_images():
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        chimerax = tomllib.load(f)['tool']['chimerax']
    assert chimerax['package-data']['src/docs/user/tools/'] == ['*.html']
    assert chimerax['extra-files']['src/docs/user/tools/' + build_help.IMAGES] == ['docs/images/*.png']


def test_help_url_falls_back_to_github_outside_chimerax():
    url = helpdocs.help_url(helpdocs.REFERENCE, 'commands')
    assert url.startswith('help:user/tools/alphalens-reference.html') or \
        url == helpdocs.ONLINE + 'docs/reference.md#commands'
