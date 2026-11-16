"""Write a minimal portable XLSX review form using standard OpenXML parts."""

from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


class ReviewWorkbook:
    """Create three-column call sheets with blank decisions and highlighted rows."""

    def write(
        self,
        path: Path,
        blocks: list[dict[str, Any]],
        feedback: dict[str, Any] | None = None,
    ) -> None:
        """Write one sheet per call, including list validation for every block."""
        calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for block in blocks:
            calls[block["recording_id"]].append(block)
        if "_review_metadata" in calls:
            raise ValueError("Reserved review metadata sheet name")
        edits = (
            {entry["identifier"]: entry for entry in feedback["entries"]}
            if feedback
            else {}
        )
        namespace = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        sheets, relations, overrides = [], [], []
        with ZipFile(path, "x", ZIP_DEFLATED) as archive:
            for number, (identifier, rows) in enumerate(calls.items(), 1):
                sheets.append(
                    f'<sheet name="{escape(identifier)}" sheetId="{number}" r:id="rId{number}"/>'
                )
                relations.append(
                    f'<Relationship Id="rId{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{number}.xml"/>'
                )
                overrides.append(
                    f'<Override PartName="/xl/worksheets/sheet{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                )
                cells = [
                    '<row r="1">'
                    + "".join(
                        f'<c r="{col}1" s="1" t="inlineStr"><is><t>{text}</t></is></c>'
                        for col, text in zip(
                            "ABCD", ["Block ID", "Decision", "Notes", "State"]
                        )
                    )
                    + "</row>"
                ]
                for index, block in enumerate(rows, 2):
                    style = 2 if block["requires_review"] else 0
                    edit = edits.get(block["identifier"], {})
                    values = [
                        block["identifier"],
                        edit.get("decision", ""),
                        edit.get("notes", ""),
                        edit.get("state", "untouched"),
                    ]
                    if any(len(value) > 32767 for value in values):
                        raise ValueError(
                            "XLSX cell exceeds 32767 characters; use JSON feedback"
                        )
                    if any(
                        ord(character) < 32 and character not in "\t\r\n"
                        for value in values
                        for character in value
                    ):
                        raise ValueError(
                            "XLSX cannot represent control characters; use JSON feedback"
                        )
                    cells.append(
                        f'<row r="{index}" ht="26" customHeight="1">'
                        + "".join(
                            f'<c r="{col}{index}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'
                            for col, value in zip("ABCD", values)
                        )
                        + "</row>"
                    )
                archive.writestr(
                    f"xl/worksheets/sheet{number}.xml",
                    f'<worksheet {namespace}><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="24" customWidth="1"/><col min="2" max="2" width="22" customWidth="1"/><col min="3" max="3" width="80" customWidth="1"/></cols><sheetData>{"".join(cells)}</sheetData><dataValidations count="1"><dataValidation type="list" allowBlank="1" showErrorMessage="1" errorTitle="Choose a decision" error="Use a value from the dropdown." sqref="B2:B{len(rows)+1}"><formula1>"Correct,Operator,Caller,Both speakers,No speech,Unsure"</formula1></dataValidation></dataValidations></worksheet>',
                )
            if feedback is not None:
                number = len(calls) + 1
                sheets.append(
                    f'<sheet name="_review_metadata" sheetId="{number}" state="hidden" r:id="rId{number}"/>'
                )
                relations.append(
                    f'<Relationship Id="rId{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{number}.xml"/>'
                )
                overrides.append(
                    f'<Override PartName="/xl/worksheets/sheet{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                )
                encoded = json.dumps(feedback)
                chunks = [encoded[i : i + 30000] for i in range(0, len(encoded), 30000)]
                cells = [
                    f'<row r="{i}"><c r="A{i}" t="inlineStr"><is><t xml:space="preserve">{escape(chunk)}</t></is></c></row>'
                    for i, chunk in enumerate(chunks, 1)
                ]
                archive.writestr(
                    f"xl/worksheets/sheet{number}.xml",
                    f'<worksheet {namespace}><sheetData>{"".join(cells)}</sheetData></worksheet>',
                )
            archive.writestr(
                "xl/workbook.xml",
                f'<workbook {namespace} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{"".join(sheets)}</sheets></workbook>',
            )
            relations.append(
                '<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(relations)
                + "</Relationships>",
            )
            archive.writestr(
                "_rels/.rels",
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="workbook" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
            )
            archive.writestr(
                "[Content_Types].xml",
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                + "".join(overrides)
                + "</Types>",
            )
            archive.writestr(
                "xl/styles.xml",
                f'<styleSheet {namespace}><fonts count="2"><font><sz val="11"/><name val="Arial"/></font><font><b/><sz val="11"/><name val="Arial"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFF0B3"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
            )

    @staticmethod
    def sheet_rows(
        archive: ZipFile, path: str, shared: list[str]
    ) -> list[dict[str, str]]:
        """Read literal cells only, including workbooks saved again by Excel."""
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        root = ET.fromstring(archive.read(path))
        rows = []
        for row in root.findall("x:sheetData/x:row", ns):
            values = {}
            for cell in row.findall("x:c", ns):
                if cell.find("x:f", ns) is not None:
                    raise ValueError("Formula cells are not accepted as feedback")
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                if column in values:
                    raise ValueError("Duplicate workbook cell")
                kind = cell.get("t")
                value = cell.findtext("x:v", default="", namespaces=ns)
                if kind == "inlineStr":
                    value = "".join(
                        item.text or "" for item in cell.findall("x:is//x:t", ns)
                    )
                elif kind == "s":
                    value = shared[int(value)]
                values[column] = value
            if any(values.values()):
                rows.append(values)
        return rows

    def read(self, path: Path) -> dict[str, Any]:
        """Recover feedback with immutable identity metadata and explicit states."""
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(path) as archive:
            if len(set(archive.namelist())) != len(archive.namelist()):
                raise ValueError("Duplicate workbook ZIP entries")
            if sum(info.file_size for info in archive.infolist()) > 20_000_000:
                raise ValueError("Feedback workbook is too large")
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(t.text or "" for t in item.findall(".//x:t", ns))
                    for item in root
                ]
            relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {r.attrib["Id"]: r.attrib["Target"] for r in relations}
            book = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = {}
            for sheet in book.findall("x:sheets/x:sheet", ns):
                name = sheet.attrib["name"]
                if name in sheets:
                    raise ValueError("Duplicate workbook sheet")
                target = targets[
                    sheet.attrib[
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                    ]
                ]
                target = (
                    target.lstrip("/") if target.startswith("/") else "xl/" + target
                )
                sheets[name] = self.sheet_rows(archive, target, shared)
        if "_review_metadata" not in sheets:
            raise ValueError("Workbook is missing review identity metadata")
        feedback = json.loads(
            "".join(row["A"] for row in sheets.pop("_review_metadata"))
        )
        entries = {e["identifier"]: copy.deepcopy(e) for e in feedback["entries"]}
        if len(entries) != len(feedback["entries"]):
            raise ValueError("Duplicate protected workbook identity")
        seen = set()
        for name, rows in sheets.items():
            if not rows or [rows[0].get(c) for c in "ABCD"] != [
                "Block ID",
                "Decision",
                "Notes",
                "State",
            ]:
                raise ValueError("Changed workbook columns")
            for row in rows[1:]:
                identifier = row.get("A", "")
                if (
                    identifier not in entries
                    or identifier in seen
                    or not identifier.startswith(name + "-B")
                ):
                    raise ValueError("Unknown, duplicate or moved workbook block")
                seen.add(identifier)
                entries[identifier].update(
                    decision=row.get("B", ""),
                    notes=row.get("C", ""),
                    state=row.get("D", ""),
                )
        if seen != set(entries):
            raise ValueError("Incomplete workbook coverage")
        feedback["entries"] = list(entries.values())
        return feedback
