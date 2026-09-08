import io
import unittest
import zipfile
import xml.etree.ElementTree as ET

from piano_clamp.prepare import (
    Candidate,
    crop_musicxml,
    direct_children,
    key_label,
    local_name,
    measure_count,
    meter_label,
    mxl_musicxml,
    select_balanced,
)


def score_xml(measures: int = 20) -> bytes:
    rows = []
    for number in range(1, measures + 1):
        attributes = ""
        if number == 1:
            attributes = """
      <attributes>
        <divisions>1</divisions><key><fifths>-3</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""
        rows.append(
            f'<measure number="{number}">{attributes}'
            '<note><pitch><step>C</step><octave>4</octave></pitch>'
            '<duration>1</duration><voice>1</voice><type>quarter</type></note></measure>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="3.1"><part-list><score-part id="P1">'
        '<part-name>Piano</part-name></score-part></part-list><part id="P1">'
        + "".join(rows)
        + "</part></score-partwise>"
    ).encode()


class PreparationTests(unittest.TestCase):
    def test_mxl_root_and_center_crop_are_self_contained(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "META-INF/container.xml",
                '<container><rootfiles><rootfile full-path="score.musicxml"/></rootfiles></container>',
            )
            archive.writestr("score.musicxml", score_xml())
        xml = mxl_musicxml(buffer.getvalue())
        cropped, source_start, source_end = crop_musicxml(
            xml, start_index=2, window_bars=16
        )
        self.assertEqual(measure_count(cropped), 16)
        self.assertEqual((source_start, source_end), ("3", "18"))
        root = ET.fromstring(cropped)
        part = direct_children(root, "part")[0]
        first_measure = direct_children(part, "measure")[0]
        attributes = direct_children(first_measure, "attributes")[0]
        names = {local_name(child) for child in list(attributes)}
        self.assertTrue({"divisions", "key", "time", "clef"}.issubset(names))
        self.assertEqual(key_label(root), "E-flat major / C minor key signature")
        self.assertEqual(meter_label(root), "3/4")

    def test_balanced_selection_prefers_parent_diversity(self):
        candidates = []
        for composer in ("Frédéric Chopin", "Wolfgang Amadeus Mozart"):
            source = "Chopin,_Frédéric" if composer.startswith("Frédéric") else "Mozart,_Wolfgang_Amadeus"
            for index, composition in enumerate(("A", "B", "C")):
                candidates.append(
                    Candidate(
                        pianocore_id=f"id-{composer}-{index}",
                        source_composer=source,
                        composer=composer,
                        composition=composition,
                        movement="",
                        score_path=f"{composer}/{composition}.mxl",
                        measure_count=20,
                    )
                )
        selected = select_balanced(
            candidates,
            target_per_composer=2,
            window_bars=16,
            seed="test",
            max_per_composition=1,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual({row.composer for row in selected}, {"Frédéric Chopin", "Wolfgang Amadeus Mozart"})
        for composer in {row.composer for row in selected}:
            self.assertEqual(len({row.composition for row in selected if row.composer == composer}), 2)


if __name__ == "__main__":
    unittest.main()
