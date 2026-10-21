"""Menus, vocabulary and the website markup must agree (no extra dependencies needed)."""
import html as _html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import vocab  # noqa: E402

HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_all_80_coco_classes_are_offered():
    names = {n for _, _, items in vocab.CATEGORIES for n in items}
    assert len(vocab.COCO_80) == 80 == len(set(vocab.COCO_80))
    assert set(vocab.COCO_80) <= names


def test_no_duplicate_names_between_menus():
    names = [n for _, _, items in vocab.CATEGORIES for n in items]
    assert len(names) == len(set(names))


def test_menus_in_html_match_vocab():
    in_html = re.findall(r'class="mm-item" data-class="([^"]+)"', HTML)
    expected = [n for _, _, items in vocab.CATEGORIES for n in items]
    assert [_html.unescape(h) for h in in_html] == expected
    assert len(re.findall(r'<div class="mm" data-cat=', HTML)) == len(vocab.CATEGORIES)


def test_universal_vocab_covers_every_menu_item():
    assert {n for _, _, items in vocab.CATEGORIES for n in items} <= set(vocab.UNIVERSAL_VOCAB)
    assert "lion" in vocab.UNIVERSAL_VOCAB  # the example that used to come out as "dog"


def test_splash_launches_after_its_css_is_defined():
    i = HTML.index("ROYAL COURT SPLASH")
    block = HTML[i:HTML.index("</script>", i)]
    assert block.rindex('if (document.body) run();') > block.index("var FRAG")
