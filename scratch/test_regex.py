import re

html = """
<tr class="oddColor">
<td align="center" class="lborder + bborder">1</td>
<td align="center" class="lborder + bborder">1</td>
<td align="center" class="lborder + bborder">2:38 / 17:22</td>
<td align="center" class="lborder + bborder">3:29 / 16:31</td>
<td align="center" class="lborder + bborder">00:51</td>
<td align="center" class="lborder + bborder + rborder">&nbsp;</td>
</tr>
"""

pat = re.compile(
    r'<td[^>]*>\s*(\d+)\s*</td>'  # Shift #
    r'\s*<td[^>]*>\s*(\d+)\s*</td>'  # Period
    r'\s*<td[^>]*>\s*(\d{1,2}:\d{2})(?:\s*/\s*\d{1,2}:\d{2})?\s*</td>'  # Start
    r'\s*<td[^>]*>\s*(\d{1,2}:\d{2})(?:\s*/\s*\d{1,2}:\d{2})?\s*</td>',
    re.DOTALL
)

m = pat.search(html)
if m:
    print(f"Match: Shift={m.group(1)}, Period={m.group(2)}, Start={m.group(3)}, End={m.group(4)}")
else:
    print("No match")
