from csv import DictReader
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("四问候选答案汇总.xlsx")
Q1 = ROOT / "data" / "processed" / "q1" / "phase4_verified"
Q23 = ROOT / "data" / "processed" / "q2q3" / "convergence_graded_v2" / "N320_dt7p5_grade2" / "samples.csv"
PAPER_RADII = (0, .5, 1, 1.5, 2)


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(DictReader(stream))


def style(sheet):
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 28
    for cell in sheet[1]:
        cell.font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000"
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(28, max(12, max(len(str(c.value or "")) for c in column) + 2))


wb = Workbook()
note = wb.active
note.title = "说明"
note.append(("项目", "内容"))
notes = (
    ("版本", "比赛候选版（误差透明），2026-09-12"),
    ("Q1", "已通过严格收敛验证；N1280，dt=0.25 s"),
    ("Q3候选终点", "206926.6772 s = 57.4796 h；中心控制"),
    ("Q3误差", "N160→N320：8.5840 s；dt 15→7.5 s：23.5400 s；保守按±60 s"),
    ("Q4候选终点", "183970.2539 s = 51.1028 h；中心控制；R≈1.200 cm"),
    ("Q4误差", "N80→N160：25.9570 s；dt 30→15 s：35.5444 s；保守按±70 s"),
    ("限制", "Q2–Q4尚未达到相邻终点差<1 s；数值来自有效模型，不是实验验证"),
)
for item in notes:
    note.append(item)

q1_times = {100, 300, 600, 900, 1200, 1500, 1800}
for title, filename in (("Q1温度", "q1_formal_temperature.csv"), ("Q1水分", "q1_formal_moisture.csv")):
    sheet = wb.create_sheet(title)
    sheet.append(("时间/s", *(f"r={r:g} cm" for r in PAPER_RADII)))
    for row in rows(Q1 / filename):
        if int(float(row["time_s"])) in q1_times:
            sheet.append((float(row["time_s"]), *(float(row[f"r_{r:g}_cm"]) for r in PAPER_RADII)))

q23 = rows(Q23)
q2 = wb.create_sheet("Q2关键场")
q2.append(("时间/h", *(f"T@{r:g}cm/℃" for r in PAPER_RADII), *(f"C@{r:g}cm/(kg/kg)" for r in PAPER_RADII)))
for row in q23:
    time = float(row["time_s"])
    if time in (1800, 3600, 5400, 7200, 9000, 10800):
        q2.append((time / 3600, *(float(row[f"temperature_r_{r:g}_cm_C"]) for r in PAPER_RADII),
                   *(float(row[f"moisture_r_{r:g}_cm"]) for r in PAPER_RADII)))

q3 = wb.create_sheet("Q3达标过程")
q3.append(("时间/h", *(f"C@{r:g}cm/(kg/kg)" for r in PAPER_RADII)))
for row in q23:
    time = float(row["time_s"])
    if time in range(21600, 194401, 21600):
        q3.append((time / 3600, *(float(row[f"moisture_r_{r:g}_cm"]) for r in PAPER_RADII)))
last = q23[-1]
q3.append((float(last["time_s"]) / 3600, *(float(last[f"moisture_r_{r:g}_cm"]) for r in PAPER_RADII)))

q4_data = (
    (6, 1.374, 1.7199106639, 1.5377129776, 1.0226453183, None, .4207526707),
    (12, 1.248, .7380613537, .6548143994, .4084068093, None, .1670278872),
    (18, 1.214, .4089701552, .3687811405, .2397928494, None, .0892201140),
    (24, 1.204, .2852465016, .2611320204, .1790105643, None, .0673505524),
    (30, 1.201, .2264093544, .2094308597, .1492807078, None, .0594985415),
    (36, 1.200, .1928786362, .1796880392, .1316710466, None, .0559691164),
    (42, 1.200, .1712844760, .1603969575, .1199852221, None, .0541093173),
    (48, 1.200, .1561861972, .1468267807, .1115685088, None, .0530162902),
    (51.11005859375, 1.200, .1499999978, .1412455575, .1080556015, None, .0526191919),
)
q4 = wb.create_sheet("Q4收缩过程")
q4.append(("时间/h", "半径/cm", "中心C", "C@0.5cm", "C@1.0cm", "C@1.5cm", "瞬时表面C"))
for item in q4_data:
    q4.append(item)

for sheet in wb.worksheets:
    style(sheet)
note.column_dimensions["B"].width = 90
wb.save(OUT)

check = load_workbook(OUT, read_only=True, data_only=False)
try:
    assert check.sheetnames == ["说明", "Q1温度", "Q1水分", "Q2关键场", "Q3达标过程", "Q4收缩过程"]
    for sheet in check.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                assert not (isinstance(cell.value, str) and (cell.value.startswith("=") or cell.value.startswith("#")))
finally:
    check.close()
print(OUT)
