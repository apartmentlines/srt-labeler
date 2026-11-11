"""Write a minimal portable XLSX review form using standard OpenXML parts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


class ReviewWorkbook:
    """Create three-column call sheets with blank decisions and highlighted rows."""

    def write(self, path: Path, blocks: list[dict[str, Any]]) -> None:
        """Write one sheet per call, including list validation for every block."""
        calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for block in blocks:
            calls[block["recording_id"]].append(block)
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
                        for col, text in zip("ABC", ["Block ID", "Decision", "Notes"])
                    )
                    + "</row>"
                ]
                for index, block in enumerate(rows, 2):
                    style = 2 if block["requires_review"] else 0
                    cells.append(
                        f'<row r="{index}" ht="26" customHeight="1"><c r="A{index}" s="{style}" t="inlineStr"><is><t>{escape(block["identifier"])}</t></is></c><c r="B{index}" s="{style}"/><c r="C{index}" s="{style}"/></row>'
                    )
                archive.writestr(
                    f"xl/worksheets/sheet{number}.xml",
                    f'<worksheet {namespace}><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="24" customWidth="1"/><col min="2" max="2" width="22" customWidth="1"/><col min="3" max="3" width="80" customWidth="1"/></cols><sheetData>{"".join(cells)}</sheetData><dataValidations count="1"><dataValidation type="list" allowBlank="1" showErrorMessage="1" errorTitle="Choose a decision" error="Use a value from the dropdown." sqref="B2:B{len(rows)+1}"><formula1>"Correct,Operator,Caller,Both speakers,No speech,Unsure"</formula1></dataValidation></dataValidations></worksheet>',
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
