"""Generate the documents from computed results, so figures cannot drift from the build.

docs/architecture.pdf       raw data -> ingestion -> analytics -> model -> dashboard, with both AI features
docs/solution_overview.pdf  what the dashboard shows, the AI features, results, insights, assumptions
README.md                   the sections between the START and END markers are filled in here
"""
from __future__ import annotations

import re

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import DOCS_DIR, GEMINI_MODEL_DEFAULT, PROCESSED_DIR, ROOT
from src.data_access import load_dq_log, load_json, load_table

INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
BLUE_WASH = colors.HexColor("#eaf2fc")
ORANGE = colors.HexColor("#eb6834")
ORANGE_WASH = colors.HexColor("#fdf0ea")
NEUTRAL = colors.HexColor("#f0efec")
LINE = colors.HexColor("#c3c2b7")


def _style(name: str, size: float, leading: float | None = None, colour=INK, bold: bool = False, **extra) -> ParagraphStyle:
    return ParagraphStyle(
        name, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
        leading=leading or size * 1.25, textColor=colour, **extra,
    )


def _box(pdf, x, y, width, height, title, lines, fill=NEUTRAL, stroke=LINE) -> None:
    """A rounded box with a bold title and wrapped text. (x, y) is its bottom-left corner.

    Text that does not fit raises an error, so nothing is silently cut off.
    """
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, 5, stroke=1, fill=1)
    pad = 7
    story = [Paragraph(title, _style("box_title", 9.8, 12, bold=True, spaceAfter=3))]
    story += [Paragraph(line, _style("box_body", 8.6, 10.6, INK_SECONDARY, spaceAfter=3)) for line in lines]
    frame = KeepInFrame(width - 2 * pad, height - 2 * pad, story, mode="error")
    _, used = frame.wrapOn(pdf, width - 2 * pad, height - 2 * pad)
    frame.drawOn(pdf, x + pad, y + height - pad - used)


def _arrow(pdf, x1, y1, x2, y2, colour=INK_SECONDARY, both_ends: bool = False) -> None:
    """A straight horizontal or vertical arrow from (x1, y1) to (x2, y2)."""
    pdf.setStrokeColor(colour)
    pdf.setFillColor(colour)
    pdf.setLineWidth(1.1)
    pdf.line(x1, y1, x2, y2)

    def head(tip_x, tip_y, from_x, from_y):
        size = 4.5
        path = pdf.beginPath()
        path.moveTo(tip_x, tip_y)
        if from_y == tip_y:
            direction = 1 if tip_x > from_x else -1
            path.lineTo(tip_x - direction * size, tip_y + size * 0.6)
            path.lineTo(tip_x - direction * size, tip_y - size * 0.6)
        else:
            direction = 1 if tip_y > from_y else -1
            path.lineTo(tip_x + size * 0.6, tip_y - direction * size)
            path.lineTo(tip_x - size * 0.6, tip_y - direction * size)
        path.close()
        pdf.drawPath(path, stroke=0, fill=1)

    head(x2, y2, x1, y1)
    if both_ends:
        head(x1, y1, x2, y2)


def build_architecture_pdf() -> None:
    report = load_json("model_report.json")
    anomaly = load_json("anomaly_report.json")
    main = report["models"]["Logistic regression"]
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(str(DOCS_DIR / "architecture.pdf"), pagesize=(page_width, page_height))
    pdf.setTitle("Credit Risk Portfolio Monitor: architecture")

    margin = 12 * mm
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(margin, page_height - margin - 6, "Credit Risk Portfolio Monitor: architecture")
    pdf.setFont("Helvetica", 8.8)
    pdf.setFillColor(INK_SECONDARY)
    pdf.drawString(
        margin, page_height - margin - 19,
        "Stages 1 to 5 run once, as a batch: python -m src.build. The hosted app reads their outputs. The only call to an outside service is to Gemini, and only when an analyst asks a question.",
    )

    # Top row: the batch pipeline, left to right.
    gap = 14
    columns = 5
    box_width = (page_width - 2 * margin - (columns - 1) * gap) / columns
    top_height = 166
    top_y = page_height - margin - 28 - top_height
    xs = [margin + i * (box_width + gap) for i in range(columns)]
    stages = [
        ("1. Raw data", ["5 CSVs in <b>data/raw/</b>, unchanged: customers, loan_accounts, bureau_credit, repayment_history, collections_events.", "<b>Tool:</b> pandas"]),
        ("2. Ingestion", ["<b>src/ingest.py, features.py</b>", "Checks keys, cleans, logs every decision, joins to one row per account. Bureau data is limited by date to avoid look-ahead.", "<b>Tools:</b> pandas, SciPy, PyArrow", "<b>Out:</b> analysis_dataset, dq_log"]),
        ("3. Analytics", ["<b>src/analytics.py</b>", "Cuts by income, employment, region, age and loan type. Writes the insights with computed figures.", "<b>Tool:</b> pandas", "<b>Out:</b> insights.json"]),
        ("4. Model", ["<b>src/model.py</b>", f"Logistic regression vs LightGBM, 5 folds grouped by customer. AUC {main['auc']:.3f}, Gini {main['gini']:.3f}, KS {main['ks']:.3f}.", "<b>Tools:</b> scikit-learn, LightGBM, SHAP", "<b>Out:</b> scores, drivers, model_report"]),
        ("5. Anomaly detection", ["<b>src/anomaly.py</b>", f"6 rule detectors and an Isolation Forest. {anomaly['accounts_flagged']} accounts have 2+ signals.", "<b>Tools:</b> pandas, scikit-learn", "<b>Out:</b> anomaly_flags"]),
    ]
    for x, (title, lines) in zip(xs, stages):
        _box(pdf, x, top_y, box_width, top_height, title, lines)
    for i in range(columns - 1):
        _arrow(pdf, xs[i] + box_width + 1, top_y + top_height / 2, xs[i + 1] - 1, top_y + top_height / 2)

    # Middle: the dashboard, which reads the outputs of stages 2 to 5.
    total_width = page_width - 2 * margin
    dash_height = 66
    dash_y = top_y - 20 - dash_height
    _box(
        pdf, margin, dash_y, total_width, dash_height,
        "6. Dashboard: app/app.py on Streamlit Community Cloud (Streamlit, Plotly). Opens without a login.",
        [
            "<b>Views:</b> Portfolio (filters, risk bands, cross-cuts, trends, insights) | Account (score, drivers, history) | Ask the Portfolio | Anomaly Review | Model &amp; Monitoring | Data Notes",
            "<b>Reads</b> the Parquet and JSON files in data/processed/, which are committed so the hosted app starts instantly. <b>Writes</b> analyst decisions and answer feedback to SQLite (src/decisions.py), with CSV export.",
        ],
        fill=BLUE_WASH, stroke=BLUE,
    )
    for x in xs[1:]:
        _arrow(pdf, x + box_width / 2, top_y - 1, x + box_width / 2, dash_y + dash_height + 1)

    # Bottom row: the two AI features, with the one outside service between them.
    bottom_y = margin
    bottom_height = dash_y - 20 - bottom_y
    gemini_width = 104
    feature_1_width = (total_width - gemini_width - 2 * gap) * 0.55
    feature_2_width = (total_width - gemini_width - 2 * gap) * 0.45
    gemini_x = margin + feature_1_width + gap
    feature_2_x = gemini_x + gemini_width + gap
    _box(
        pdf, margin, bottom_y, feature_1_width, bottom_height,
        "AI feature 1: Ask the Portfolio (src/query_engine.py, src/assistant.py)",
        [
            "<b>Step 1. Route.</b> Gemini reads the question and must call query_portfolio, get_account or decline.",
            "<b>Step 2. Compute.</b> Code validates that choice against a fixed list of metrics and dimensions, and pandas computes the result. There is no SQL and no generated code.",
            "<b>Step 3. Phrase.</b> Gemini words the result in 1-3 sentences.",
            "<b>Step 4. Check.</b> Code confirms every number appears in the result. If one does not, a code-written answer replaces it. The source line is always written by code.",
            "<b>Sent to Gemini.</b> Step 1: the question and the catalogue of metric, dimension and value names, with no data rows. Step 3: the question and the computed result (aggregates, or one account's score, band, drivers and flags).",
            "<b>Never sent:</b> customer IDs or raw tables. The API key stays in the host's secrets store.",
            "<b>Without a key, or on a 429:</b> a guided dropdown query uses the same calculations with no LLM.",
        ],
        fill=ORANGE_WASH, stroke=ORANGE,
    )
    _box(
        pdf, gemini_x, bottom_y + bottom_height / 2 - 108, gemini_width, 216,
        "Gemini API",
        [f"{GEMINI_MODEL_DEFAULT}", "Outside service. 2 calls per question, forced function call on the first. If a model is busy, the next one is tried.", "Free tier may use content to improve Google's products: fine for synthetic data, not for production."],
        fill=colors.white, stroke=ORANGE,
    )
    _arrow(pdf, margin + feature_1_width + 1, bottom_y + bottom_height / 2, gemini_x - 1, bottom_y + bottom_height / 2, colour=ORANGE, both_ends=True)
    _box(
        pdf, feature_2_x, bottom_y, feature_2_width, bottom_height,
        "AI feature 2: Anomaly Review (src/anomaly.py, src/decisions.py)",
        [
            "<b>No LLM, and no data leaves the app.</b>",
            "<b>Own history:</b> consecutive missed payments, sudden deterioration, clean record then a miss.",
            "<b>Peer segment:</b> far behind peers, high bureau debt for the income band.",
            "<b>Bureau and overall:</b> multiple write-offs, and an Isolation Forest across all signals.",
            "An account joins the ranked review list when 2+ agree. The analyst sees why, with the account's own figures, and records Confirm or Dismiss with a note. Dismiss rates per detector feed threshold reviews.",
            f"target_default is never used to fit a detector. As a sanity check only, the list defaults at {anomaly['flagged_default_rate']:.1%} against {anomaly['base_default_rate']:.1%} overall.",
        ],
        fill=ORANGE_WASH, stroke=ORANGE,
    )
    _arrow(pdf, margin + feature_1_width / 2, dash_y - 1, margin + feature_1_width / 2, bottom_y + bottom_height + 1, both_ends=True)
    _arrow(pdf, feature_2_x + feature_2_width / 2, dash_y - 1, feature_2_x + feature_2_width / 2, bottom_y + bottom_height + 1, both_ends=True)

    pdf.showPage()
    pdf.save()


def build_overview_pdf() -> None:
    report = load_json("model_report.json")
    anomaly = load_json("anomaly_report.json")
    insights = load_json("insights.json")
    logistic = report["models"]["Logistic regression"]
    lightgbm = report["models"]["LightGBM"]
    deciles = report["deciles"]
    sensitivity = {row["variant"]: row for row in report["sensitivity"]}
    collections_before_loan = load_table("collections_clean")["event_before_origination"].mean()

    margin = 12 * mm
    document = SimpleDocTemplate(
        str(DOCS_DIR / "solution_overview.pdf"), pagesize=A4, leftMargin=margin, rightMargin=margin,
        topMargin=9 * mm, bottomMargin=8 * mm, title="Credit Risk Portfolio Monitor: solution overview",
    )
    title = _style("title", 14, bold=True, spaceAfter=2)
    heading = _style("heading", 9.6, 11.5, BLUE, bold=True, spaceBefore=5, spaceAfter=1.5)
    body = _style("body", 8.1, 9.8, INK_SECONDARY)
    bullet = _style("bullet", 8.1, 9.8, INK_SECONDARY, leftIndent=8, bulletIndent=0, spaceAfter=1.2)
    cell = _style("cell", 7.7, 9.2, INK_SECONDARY)
    cell_head = _style("cell_head", 7.7, 9.2, INK, bold=True)

    def bullets(items):
        return [Paragraph(item, bullet, bulletText="\u2022") for item in items]

    def grid(rows, widths):
        data = [[Paragraph(str(c), cell_head if r == 0 or i == 0 else cell) for i, c in enumerate(row)] for r, row in enumerate(rows)]
        table = Table(data, colWidths=widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NEUTRAL),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
                ]
            )
        )
        return table

    width = A4[0] - 2 * margin
    story = [
        Paragraph("Credit Risk Portfolio Monitor: solution overview", title),
        Paragraph(
            f"A hosted dashboard for a risk analyst, built on 5 synthetic bank tables ({report['accounts']:,} loan accounts, {report['base_default_rate']:.1%} default rate), with an LLM assistant and anomaly detection embedded.",
            body,
        ),
        Paragraph("What the dashboard shows and how an analyst uses it", heading),
        grid(
            [
                ["View", "What it shows", "How an analyst uses it"],
                ["Portfolio", "KPIs, insights, exposure and default by risk band, any dimension or two-way cross-cut, vintage and delinquency trends, riskiest accounts.", "Filter to a segment, find where risk and money concentrate, select a row to drill into an account."],
                ["Account", "Score, risk band, percentile, key drivers, 12-month repayment against peers, bureau lines, collections, data flags.", "Understand why one account scores as it does before acting on it."],
                ["Ask the Portfolio", "Plain-English questions answered with exact figures and a source. A guided query needs no LLM.", "Get a figure without building a cut, and check how it was computed."],
                ["Anomaly Review", f"Ranked list of {anomaly['accounts_flagged']} accounts with reasons, peer comparison, decision form and log.", "Work the list: read why, then Confirm or Dismiss with a note."],
                ["Model & Monitoring", "AUC, KS, Gini, ROC, deciles, SHAP, sensitivity, PSI, fairness, monitoring plan.", "Judge how far to trust the score, and spot drift."],
                ["Data Notes", "Join logic, every data-quality decision with counts, assumptions.", "Check how the numbers were made."],
            ],
            [width * 0.15, width * 0.48, width * 0.37],
        ),
        Paragraph("The two AI features", heading),
        grid(
            [
                ["", "Ask the Portfolio (Gemini)", "Anomaly detection"],
                ["How it works", "Gemini only picks a query from a fixed catalogue, or declines. pandas computes the figures. Gemini then words the result, and code checks every number, replacing the wording if one is unsupported.", "Six rule detectors and an Isolation Forest compare each account with its own history, its peers and its bureau record. The target is never used to fit them."],
                ["Trigger", "An analyst submits a question.", f"Batch run. An account is listed when 2+ detectors agree: {anomaly['accounts_flagged']} accounts, {anomaly['share_flagged']:.1%}."],
                ["False positive", "Answers what it should decline, or reads the question as a different metric, such as originated amount for balance.", f"A one-off payment slip, agreed hardship plan or data error. {1 - anomaly['flagged_default_rate']:.0%} of listed accounts did not default."],
                ["False negative", "Declines an answerable question because the wording maps to no listed metric.", f"Steadily poor but normal for peers, or a clean-record defaulter. {1 - anomaly['share_of_defaults_in_list']:.0%} of defaults are not listed: it does not replace the model."],
                ["Human step", "The analyst checks the query and source under each answer and can mark it as wrong. Nothing changes as a result.", "The analyst records Confirm or Dismiss with a note. Nothing reaches a customer without it. Dismiss rates tune thresholds."],
            ],
            [width * 0.12, width * 0.46, width * 0.42],
        ),
        Paragraph("Model results and approach", heading),
        grid(
            [
                ["Model", "AUC (95% interval)", "Gini", "KS", "Out-of-time AUC"],
                ["Logistic regression (main)", f"{logistic['auc']:.3f} ({logistic['auc_95_interval'][0]:.3f}-{logistic['auc_95_interval'][1]:.3f})", f"{logistic['gini']:.3f}", f"{logistic['ks']:.3f}", f"{logistic['out_of_time']['auc']:.3f}"],
                ["LightGBM (challenger)", f"{lightgbm['auc']:.3f} ({lightgbm['auc_95_interval'][0]:.3f}-{lightgbm['auc_95_interval'][1]:.3f})", f"{lightgbm['gini']:.3f}", f"{lightgbm['ks']:.3f}", f"{lightgbm['out_of_time']['auc']:.3f}"],
            ],
            [width * 0.32, width * 0.26, width * 0.12, width * 0.12, width * 0.18],
        ),
        Spacer(1, 2),
        Paragraph(
            f"{report['feature_count']} features from customer and loan data, 12 months of repayment and date-limited bureau history, validated with 5 folds grouped by customer. LightGBM had to win by 0.005 AUC to replace the simpler model, and did not. "
            f"The riskiest 10% of scores default at {deciles[0]['default_rate']:.1%}, and the riskiest 30% hold {deciles[2]['cumulative_capture']:.0%} of defaults. "
            f"Bureau data is cut off at the scoring date (ignoring dates would give AUC {sensitivity['Bureau lines ignoring dates (look-ahead)']['auc']:.3f}). "
            f"Gender, marital status, region and age are excluded pending compliance review (age alone would add {sensitivity['Final model + age']['auc'] - logistic['auc']:.3f}). "
            "The signal is modest by nature: a much higher AUC would suggest leakage. After deployment: monthly PSI on scores and features, quarterly AUC, KS, calibration and fairness checks by vintage, with set retraining triggers.",
            body,
        ),
        Paragraph("Top 3 portfolio insights", heading),
    ]
    story += bullets([f"<b>{i['title']}.</b> {i['finding']} <b>So what:</b> {i['so_what']}" for i in insights if i["top3"]])
    story += [
        Paragraph("Key assumptions", heading),
        Paragraph(
            "The target has no date, so default is assumed to come after month 12, making this a month-12 behaviour score (3- and 6-month sensitivity is reported). The data is assumed to be a 2025-12-31 snapshot. Status and balances on active bureau lines "
            "have no reporting date. Latest balance is approximate, because it falls by a fixed 60% of each payment. Free hosting can reset the decision log on restart.",
            body,
        ),
        Paragraph("What production would need from the bank's IT and data teams", heading),
    ]
    story += bullets(
        [
            f"<b>Data:</b> a dated default flag and definition (such as 90+ days past due), collateral values and recoveries to size loss, bureau data with reporting dates, and calendar dates on repayments. Collections data needs fixing: {collections_before_loan:.0%} of events pre-date their loan.",
            "<b>Platform:</b> automated feeds, a governed database for decisions with analyst sign-in and an audit trail, access control, a model registry and secrets management.",
            "<b>LLM and governance:</b> an approved enterprise LLM provider where content is not used for training, with data-residency review and spend limits. Model validation, and compliance review of fair-lending choices, especially age.",
        ]
    )
    document.build(story)


def _replace_section(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.S)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {name} markers")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text)


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    def clean(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def update_readme() -> None:
    """Fill in the README sections that come from computed results."""
    path = ROOT / "README.md"
    if not path.exists():
        return
    report = load_json("model_report.json")
    anomaly = load_json("anomaly_report.json")
    insights = load_json("insights.json")
    log = load_dq_log()

    models = _markdown_table(
        ["Model", "AUC", "95% interval", "Gini", "KS", "Out-of-time AUC", "Out-of-time KS"],
        [
            [
                f"{name} ({'main' if name == report['champion'] else 'challenger'})",
                f"{m['auc']:.3f}", f"{m['auc_95_interval'][0]:.3f} to {m['auc_95_interval'][1]:.3f}", f"{m['gini']:.3f}", f"{m['ks']:.3f}",
                f"{m['out_of_time']['auc']:.3f}", f"{m['out_of_time']['ks']:.3f}",
            ]
            for name, m in report["models"].items()
        ],
    )
    final_auc = report["sensitivity"][0]["auc"]
    sensitivity = _markdown_table(
        ["Variant", "AUC", "Change", "Why it is here"],
        [[row["variant"], f"{row['auc']:.3f}", f"{row['auc'] - final_auc:+.3f}", row["note"]] for row in report["sensitivity"]],
    )
    insight_text = "\n".join(
        f"{position}. **{item['title']}.** {item['finding']} *So what:* {item['so_what']}"
        for position, item in enumerate(insights, start=1)
    )
    detectors = _markdown_table(
        ["Detector", "Compared with", "Rule", "Accounts", "In review list", "Default rate"],
        [[d["name"], d["compared_with"], d["rule"], f"{d['accounts']:,}", f"{d['in_review_list']:,}", f"{d['default_rate']:.1%}"] for d in anomaly["detectors"]],
    )
    anomaly_summary = (
        f"{anomaly['accounts_flagged']:,} accounts ({anomaly['share_flagged']:.1%}) are in the review list, with a default rate of "
        f"{anomaly['flagged_default_rate']:.1%} against {anomaly['base_default_rate']:.1%} overall. That is a sanity check only: the target is never used to fit a detector.\n\n{detectors}"
    )
    quality = _markdown_table(
        ["Table", "Column", "Issue", "Rows", "Decision", "In dictionary"],
        [[row.table, row.column, row.issue, f"{row.rows_affected:,}", row.decision, "Yes" if row.in_dictionary else "No"] for row in log.itertuples()],
    )

    live_test = "Not run yet."
    evaluation_path = PROCESSED_DIR / "assistant_eval.json"
    if evaluation_path.exists():
        evaluation = load_json("assistant_eval.json")
        live_test = (
            f"Latest live test, with {evaluation['model']} as the main model: **{evaluation['answerable_correct']} of {evaluation['answerable_total']}** answerable questions "
            f"matched a direct calculation, **{evaluation['unanswerable_declined']} of {evaluation['unanswerable_total']}** unanswerable questions were declined, and the number check "
            f"replaced the wording of {evaluation['wording_blocked_by_number_check']} answers. Full results are in `data/processed/assistant_eval.json`."
        )

    text = path.read_text(encoding="utf-8")
    for name, content in [
        ("LIVE_TEST", live_test),
        ("MODEL_RESULTS", models),
        ("SENSITIVITY", sensitivity),
        ("INSIGHTS", insight_text),
        ("ANOMALY", anomaly_summary),
        ("DATA_QUALITY", quality),
    ]:
        text = _replace_section(text, name, content)
    path.write_text(text, encoding="utf-8")


def build_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    build_architecture_pdf()
    build_overview_pdf()
    for name in ("architecture.pdf", "solution_overview.pdf"):
        pages = len(PdfReader(DOCS_DIR / name).pages)
        if pages != 1:
            raise ValueError(f"{name} has {pages} pages, and the brief allows exactly one")
    update_readme()


if __name__ == "__main__":
    build_docs()
    print(f"Wrote the PDFs to {DOCS_DIR} and updated the README")
